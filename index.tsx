import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import * as THREE from 'three';
// We rely on the importmap to resolve 'three/examples/jsm/...' to the CDN URL
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

// --- Configuration & Constants ---
const ORNAMENT_COUNT = 550; // Increased count slightly
const TREE_HEIGHT = 28;
const TREE_BASE_RADIUS = 12;
const COLORS = {
  GOLD: 0xFFD700,
  RED: 0xC41E3A, // Darker, richer red
  GREEN: 0x0B4619, // Deep forest green
  WARM_WHITE: 0xFFFDD0
};

// --- Helper Functions ---

// Generate a position on a cone spiral (improved shape)
const getTreePosition = (index: number, total: number) => {
  // Use a power function for height to make the tree denser at the bottom
  const normalizedIndex = index / total;
  const y = (1 - normalizedIndex) * TREE_HEIGHT - (TREE_HEIGHT / 2); 
  
  // Radius tapers linearly as we go up
  const radius = (normalizedIndex) * TREE_BASE_RADIUS;
  
  // Golden angle for perfect organic spiral distribution
  const angle = index * 2.39996; 
  
  const x = Math.cos(angle) * radius;
  const z = Math.sin(angle) * radius;
  return new THREE.Vector3(x, y, z);
};

// Generate a random position in a box volume
const getFloatPosition = () => {
  const range = 45;
  return new THREE.Vector3(
    (Math.random() - 0.5) * range,
    (Math.random() - 0.5) * range,
    (Math.random() - 0.5) * range
  );
};

// --- Procedural Texture Generators ---

const createNoiseTexture = (intensity = 1.0) => {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) return new THREE.Texture();
  
  const imageData = ctx.createImageData(size, size);
  const data = imageData.data;
  
  for (let i = 0; i < data.length; i += 4) {
    const val = Math.random() * 255 * intensity;
    data[i] = val;     // r
    data[i + 1] = val; // g
    data[i + 2] = val; // b
    data[i + 3] = 255; // alpha
  }
  
  ctx.putImageData(imageData, 0, 0);
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  return texture;
};

const createStripeTexture = () => {
  const size = 512;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) return new THREE.Texture();

  // White background
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, size, size);

  // Red stripes
  ctx.fillStyle = '#C41E3A';
  const stripeCount = 8;
  const stripeWidth = size / stripeCount;
  
  // Diagonal rotation
  ctx.translate(size/2, size/2);
  ctx.rotate(Math.PI / 4);
  ctx.translate(-size, -size);

  for(let i = 0; i < stripeCount * 3; i++) {
     ctx.fillRect(i * stripeWidth * 2, -size, stripeWidth, size * 4);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
};

// --- Components ---

const App = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // App State
  const [loading, setLoading] = useState(true);
  const [started, setStarted] = useState(false);
  const [gesture, setGesture] = useState<string>('检测中...');
  const [userPhotos, setUserPhotos] = useState<string[]>([]);
  
  // Refs for Three.js & Logic
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const composerRef = useRef<EffectComposer | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const particlesRef = useRef<any[]>([]); // Stores mesh and target data
  const photoMeshesRef = useRef<THREE.Mesh[]>([]);
  const starRef = useRef<THREE.Mesh | null>(null); // The star on top
  const frameIdRef = useRef<number>(0);
  
  // Logic Refs
  const stateRef = useRef<'TREE' | 'FLOAT' | 'FOCUS'>('TREE');
  const focusedPhotoIndex = useRef<number>(-1);
  const handRotationRef = useRef<{x: number, y: number}>({ x: 0, y: 0 });
  
  // Gesture Smoothing (Debounce)
  const gestureHistoryRef = useRef<string[]>([]);
  const GESTURE_HISTORY_LIMIT = 5; 

  // Initialize MediaPipe
  useEffect(() => {
    const initVision = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
        );
        handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1
        });
        setLoading(false);
      } catch (error: any) {
        console.error("Failed to load MediaPipe:", error);
        alert(`AI 模型加载失败: ${error.message}\n请检查网络连接 (可能需要访问 Google 服务)`);
      }
    };
    initVision();
  }, []);

  // Initialize Three.js
  useEffect(() => {
    if (!mountRef.current) return;

    // Scene
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x050805, 0.015); // Deep green-black fog
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 45);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Post Processing (Bloom)
    const renderScene = new RenderPass(scene, camera);
    const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.5, 0.4, 0.85);
    bloomPass.threshold = 0.15;
    bloomPass.strength = 1.8; // Stronger glow for "Cinematic" feel
    bloomPass.radius = 0.8;
    
    const composer = new EffectComposer(renderer);
    composer.addPass(renderScene);
    composer.addPass(bloomPass);
    composerRef.current = composer;

    // Lights
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(COLORS.GOLD, 2, 100);
    pointLight.position.set(10, 10, 20);
    scene.add(pointLight);
    
    // Bottom red glow
    const pointLight2 = new THREE.PointLight(COLORS.RED, 3, 80);
    pointLight2.position.set(-10, -20, 10);
    scene.add(pointLight2);
    
    // Top-down spotlight for drama
    const spotLight = new THREE.SpotLight(0xfffae6, 8);
    spotLight.position.set(0, 60, 0);
    spotLight.angle = Math.PI / 5;
    spotLight.penumbra = 0.5;
    spotLight.castShadow = true;
    scene.add(spotLight);

    // --- Materials & Textures ---
    
    const noiseTexture = createNoiseTexture(1.0);
    const lightNoiseTexture = createNoiseTexture(0.3);
    const stripeTexture = createStripeTexture();
    
    // Gold: Roughness map makes it look like old foil/metal
    const materialGold = new THREE.MeshStandardMaterial({ 
      color: COLORS.GOLD, 
      metalness: 1.0, 
      roughness: 0.4, 
      roughnessMap: noiseTexture,
      emissive: 0x332200 
    });

    // Red: Slight bump map for "glitter" or matte finish
    const materialRed = new THREE.MeshStandardMaterial({ 
      color: COLORS.RED, 
      metalness: 0.6, 
      roughness: 0.3, 
      bumpMap: lightNoiseTexture,
      bumpScale: 0.01,
      emissive: 0x220000 
    });

    // Green: Shiny but textured
    const materialGreen = new THREE.MeshStandardMaterial({ 
      color: COLORS.GREEN, 
      metalness: 0.4, 
      roughness: 0.7,
      bumpMap: lightNoiseTexture,
      bumpScale: 0.02
    });

    const materialWhite = new THREE.MeshStandardMaterial({
        color: COLORS.WARM_WHITE, metalness: 0.1, roughness: 0.1, emissive: 0x555555
    });

    // Candy Cane Material
    const materialCandy = new THREE.MeshStandardMaterial({
        map: stripeTexture,
        roughness: 0.3,
        metalness: 0.1
    });

    // --- Geometry Construction ---

    const geometrySphere = new THREE.SphereGeometry(0.6, 24, 24); // Higher detail for bump maps
    const geometryBox = new THREE.BoxGeometry(0.8, 0.8, 0.8);
    const geometryCandyCane = new THREE.CylinderGeometry(0.15, 0.15, 1.5, 8); // Candy sticks

    const particles: any[] = [];

    for (let i = 0; i < ORNAMENT_COUNT; i++) {
      const rand = Math.random();
      let geo, mat;

      if (rand < 0.05) {
          geo = geometryCandyCane;
          mat = materialCandy;
      } else if (rand < 0.15) {
          geo = geometryBox; // Presents
          mat = materialRed;
      } else if (rand < 0.4) {
          geo = geometrySphere;
          mat = materialGold;
      } else if (rand < 0.65) {
          geo = geometrySphere;
          mat = materialRed;
      } else if (rand < 0.95) {
          geo = geometrySphere; // Filler greenery
          mat = materialGreen;
      } else {
          geo = geometrySphere; // Lights
          mat = materialWhite;
      }

      const mesh = new THREE.Mesh(geo, mat);
      
      const treePos = getTreePosition(i, ORNAMENT_COUNT);
      const floatPos = getFloatPosition();

      mesh.position.copy(treePos);
      
      // Random rotation
      mesh.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, 0);
      
      const scale = 0.5 + Math.random() * 0.8;
      mesh.scale.set(scale, scale, scale);

      mesh.userData = {
        treePos: treePos,
        floatPos: floatPos,
        rotationSpeed: new THREE.Vector3(
            (Math.random() - 0.5) * 0.02, 
            (Math.random() - 0.5) * 0.02, 
            (Math.random() - 0.5) * 0.02
        ),
        phase: Math.random() * Math.PI * 2 // For twinkling
      };

      scene.add(mesh);
      particles.push(mesh);
    }
    particlesRef.current = particles;

    // 2. The Star on Top
    const starGeo = new THREE.IcosahedronGeometry(2, 0);
    const starMat = new THREE.MeshBasicMaterial({ color: 0xffffee });
    const starMesh = new THREE.Mesh(starGeo, starMat);
    starMesh.position.set(0, TREE_HEIGHT/2 + 2, 0);
    scene.add(starMesh);
    starRef.current = starMesh;

    // Handle Resize
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
      composer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Handle Photo Uploads
  useEffect(() => {
    if (!sceneRef.current) return;

    const loader = new THREE.TextureLoader();
    
    // Clear old photos
    photoMeshesRef.current.forEach(m => sceneRef.current?.remove(m));
    photoMeshesRef.current = [];
    
    userPhotos.forEach((url, index) => {
      loader.load(url, (texture) => {
        texture.colorSpace = THREE.SRGBColorSpace; // Correct color
        const aspect = texture.image.width / texture.image.height;
        const geo = new THREE.PlaneGeometry(4 * aspect, 4);
        const mat = new THREE.MeshBasicMaterial({ 
          map: texture, 
          side: THREE.DoubleSide,
        });

        // Define the mesh for the photo
        const mesh = new THREE.Mesh(geo, mat);
        
        // Add a gold border frame
        const frameGeo = new THREE.PlaneGeometry(4 * aspect + 0.2, 4 + 0.2);
        // Reuse gold material but maybe clone it to avoid texture mapping issues on plain geometry if needed
        // For simple frame, standard material is fine
        const frameMat = new THREE.MeshStandardMaterial({ color: COLORS.GOLD, metalness: 1.0, roughness: 0.3 });
        const frameMesh = new THREE.Mesh(frameGeo, frameMat);
        frameMesh.position.z = -0.05; // Slightly behind photo
        
        const group = new THREE.Group();
        group.add(mesh);
        group.add(frameMesh);
        
        // Calculate Positions
        // Distribute photos spirally but further out
        const totalItems = userPhotos.length;
        const y = (index / totalItems) * (TREE_HEIGHT * 0.8) - (TREE_HEIGHT * 0.4);
        const angle = index * (Math.PI * 2 / 1.618); // Golden ratio steps
        const radius = TREE_BASE_RADIUS * 0.8;
        
        const treePos = new THREE.Vector3(
            Math.cos(angle) * radius,
            y,
            Math.sin(angle) * radius
        );
        treePos.multiplyScalar(1.3); // Push out of foliage
        
        const floatPos = getFloatPosition();
        
        group.position.copy(treePos);
        group.lookAt(0, 0, 0);
        
        // Store user data on the group
        group.userData = {
          treePos: treePos,
          floatPos: floatPos,
          isPhoto: true,
          originalScale: new THREE.Vector3(1, 1, 1),
          phase: Math.random() * Math.PI
        };
        
        // Needed for tracking
        // We'll cast to Mesh just for TypeScript array compatibility or wrapper it
        const groupAsMesh = group as unknown as THREE.Mesh;
        
        sceneRef.current?.add(group);
        photoMeshesRef.current.push(groupAsMesh);
      });
    });

  }, [userPhotos]);

  // Main Loop
  const animate = useCallback(() => {
    if (!started || !cameraRef.current || !sceneRef.current) return;

    // 1. Detect Hand & Smooth Gestures
    let currentFrameGesture: 'TREE' | 'FLOAT' | 'FOCUS' | null = null;
    let handPos = { x: 0, y: 0 }; 

    if (handLandmarkerRef.current && videoRef.current && videoRef.current.currentTime > 0) {
      try {
        const results = handLandmarkerRef.current.detectForVideo(videoRef.current, Date.now());
        
        if (results.landmarks && results.landmarks.length > 0) {
          const landmarks = results.landmarks[0];
          const wrist = landmarks[0];
          
          // Thumb: 1, 2, 3, 4
          const thumbTip = landmarks[4];
          const thumbIP = landmarks[3];
          
          // Index: 5, 6, 7, 8
          const indexTip = landmarks[8];
          
          // Map Hand Position
          handPos.x = (wrist.x - 0.5) * 2;
          handPos.y = (wrist.y - 0.5) * 2;

          // --- Improved Gesture Detection ---
          
          // Check openness of 4 fingers (Index, Middle, Ring, Pinky)
          // Compare Tip distance to wrist vs PIP (joint 2) distance to wrist
          // If Tip is further, it's open. If Tip is closer, it's curled.
          const fingerTips = [8, 12, 16, 20];
          const fingerPIPs = [6, 10, 14, 18]; // Joints: 5-8, 9-12... 6 is PIP
          let openCount = 0;
          
          for(let i=0; i<4; i++) {
              const tip = landmarks[fingerTips[i]];
              const pip = landmarks[fingerPIPs[i]];
              
              const distTip = Math.hypot(tip.x - wrist.x, tip.y - wrist.y);
              const distPip = Math.hypot(pip.x - wrist.x, pip.y - wrist.y);
              
              // 1.1 multiplier provides a small buffer
              if (distTip > distPip * 1.1) {
                  openCount++;
              }
          }
          
          // Check Thumb
          // Thumb is open if tip is far from Index MCP (5)
          const indexMCP = landmarks[5];
          const thumbDistToIndex = Math.hypot(thumbTip.x - indexMCP.x, thumbTip.y - indexMCP.y);
          if (thumbDistToIndex > 0.15) openCount++; // Rough heuristic for thumb

          // Pinch logic (Thumb tip close to Index tip)
          const pinchDist = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);

          // Priority Logic
          if (pinchDist < 0.05) {
            currentFrameGesture = 'FOCUS';
          } else if (openCount <= 1) { 
            // Fist: 0 or 1 finger open (sometimes thumb is tricky)
            currentFrameGesture = 'TREE';
          } else if (openCount >= 3) { 
            // Open: 3 or more fingers open
            currentFrameGesture = 'FLOAT';
            handRotationRef.current = { x: handPos.x, y: handPos.y };
          }
        }
      } catch (err) {
        // Suppress transient detection errors
      }
    }

    // Debounce Logic
    if (currentFrameGesture) {
        gestureHistoryRef.current.push(currentFrameGesture);
    } 
    
    if (gestureHistoryRef.current.length > GESTURE_HISTORY_LIMIT) {
        gestureHistoryRef.current.shift();
    }

    // Determine dominant gesture in history
    const counts = { TREE: 0, FLOAT: 0, FOCUS: 0 };
    gestureHistoryRef.current.forEach(g => {
        if (g in counts) counts[g as keyof typeof counts]++;
    });

    let detectedState: 'TREE' | 'FLOAT' | 'FOCUS' | null = null;
    const threshold = Math.ceil(GESTURE_HISTORY_LIMIT * 0.6); // Majority vote

    if (counts.FOCUS >= threshold) detectedState = 'FOCUS';
    else if (counts.TREE >= threshold) detectedState = 'TREE';
    else if (counts.FLOAT >= threshold) detectedState = 'FLOAT';

    // Update UI text
    if (detectedState === 'TREE') setGesture('✊ 握拳 (聚合圣诞树)');
    else if (detectedState === 'FLOAT') setGesture('🖐 张开 (漂浮 & 旋转)');
    else if (detectedState === 'FOCUS') setGesture('🤏 捏合 (查看照片)');
    else if (!currentFrameGesture) setGesture('未检测到手势');

    // State Transition
    if (detectedState) {
       if (detectedState === 'FOCUS') {
         if (stateRef.current !== 'FOCUS') {
             stateRef.current = 'FOCUS';
             if (photoMeshesRef.current.length > 0) {
               focusedPhotoIndex.current = (focusedPhotoIndex.current + 1) % photoMeshesRef.current.length;
             }
         }
       } else {
         stateRef.current = detectedState;
         focusedPhotoIndex.current = -1;
       }
    }

    // 3. Animation
    const time = Date.now() * 0.001;
    const currentState = stateRef.current;

    // Star Animation
    if (starRef.current) {
        starRef.current.rotation.y = time * 0.5;
        starRef.current.rotation.z = Math.sin(time) * 0.1;
        const starScale = 1 + Math.sin(time * 3) * 0.1;
        starRef.current.scale.set(starScale, starScale, starScale);
        
        // Move star based on state
        if (currentState === 'TREE') {
            starRef.current.position.lerp(new THREE.Vector3(0, TREE_HEIGHT/2 + 1, 0), 0.05);
        } else {
            starRef.current.position.lerp(new THREE.Vector3(0, 20, 0), 0.05);
        }
    }

    // Particles Animation
    particlesRef.current.forEach((mesh) => {
      let target;
      if (currentState === 'TREE') {
        target = mesh.userData.treePos;
      } else {
        target = mesh.userData.floatPos;
      }
      
      mesh.position.lerp(target, 0.04); // Smoother lerp
      
      mesh.rotation.x += mesh.userData.rotationSpeed.x;
      mesh.rotation.y += mesh.userData.rotationSpeed.y;

      // Twinkle effect (Scale pulsing)
      const twinkle = Math.sin(time * 2 + mesh.userData.phase) * 0.1 + 1.0;
      mesh.scale.setScalar(twinkle * (mesh === starRef.current ? 2 : 1) * (currentState === 'FLOAT' ? 0.8 : 1.0));
    });

    // Photos Animation
    photoMeshesRef.current.forEach((group, i) => {
      const mesh = group as unknown as THREE.Group; // It's actually a Group
      let targetPos = new THREE.Vector3();
      let targetScale = new THREE.Vector3(1, 1, 1);
      let targetRot = new THREE.Quaternion();

      if (currentState === 'TREE') {
        targetPos.copy(group.userData.treePos);
        // Look away from center
        const lookAtPos = mesh.position.clone().multiplyScalar(2);
        const m = new THREE.Matrix4();
        m.lookAt(lookAtPos, mesh.position, new THREE.Vector3(0, 1, 0));
        targetRot.setFromRotationMatrix(m);
        targetScale.setScalar(0.8); // Smaller on tree

      } else if (currentState === 'FOCUS' && i === focusedPhotoIndex.current) {
        targetPos.set(0, 0, 25);
        targetScale.set(3.5, 3.5, 3.5);
        targetRot.setFromEuler(new THREE.Euler(0, 0, 0));
      } else {
        targetPos.copy(group.userData.floatPos);
        targetRot.setFromEuler(new THREE.Euler(
            Math.sin(time * 0.1 + i) * 0.5, 
            time * 0.05, 
            0
        ));
      }

      mesh.position.lerp(targetPos, 0.06);
      mesh.scale.lerp(targetScale, 0.06);
      mesh.quaternion.slerp(targetRot, 0.06);
    });

    // Camera Logic
    if (currentState === 'FLOAT' || currentState === 'FOCUS') {
      const angle = handRotationRef.current.x * Math.PI; 
      const height = handRotationRef.current.y * 15;
      
      const r = 45;
      const targetCamX = Math.sin(angle) * r;
      const targetCamZ = Math.cos(angle) * r;
      const targetCamY = -height; 

      cameraRef.current.position.lerp(new THREE.Vector3(targetCamX, targetCamY, targetCamZ), 0.04);
      cameraRef.current.lookAt(0, 0, 0);
    } else {
      cameraRef.current.position.lerp(new THREE.Vector3(0, 2, 45), 0.04);
      cameraRef.current.lookAt(0, 5, 0); // Look slightly up at tree center
    }

    composerRef.current?.render();
    frameIdRef.current = requestAnimationFrame(animate);
  }, [started]);


  // Loop Starter
  useEffect(() => {
    if (started) {
      frameIdRef.current = requestAnimationFrame(animate);
    }
    return () => cancelAnimationFrame(frameIdRef.current);
  }, [started, animate]);


  const handleStart = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert("浏览器不支持或当前环境不安全。请尝试使用 Chrome/Safari 并确保使用 HTTPS。");
      return;
    }

    try {
      const constraints = {
        video: { 
          facingMode: "user",
          width: { ideal: 640 }, // Lower resolution for better performance
          height: { ideal: 480 } 
        } 
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        
        // Ensure the video is playing before MediaPipe tries to read it
        videoRef.current.onloadeddata = () => {
           videoRef.current?.play().catch(e => console.error("Video play error", e));
           setStarted(true);
        };
      }
    } catch (err: any) {
      console.error("Camera access denied:", err);
      let msg = "无法访问摄像头：";
      if (err.name === 'NotAllowedError') {
        msg += "请在浏览器设置中开启摄像头权限并刷新页面。";
      } else if (err.name === 'NotFoundError') {
        msg += "未检测到摄像头设备。";
      } else if (err.name === 'NotReadableError') {
        msg += "摄像头可能被其他应用占用。";
      } else {
        msg += err.message;
      }
      alert(msg);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newPhotos: string[] = [];
      // Loop through all selected files
      Array.from(e.target.files).forEach(file => {
          newPhotos.push(URL.createObjectURL(file));
      });
      setUserPhotos(prev => [...prev, ...newPhotos]);
    }
  };
  
  const handleClearPhotos = () => {
      setUserPhotos([]);
  }

  return (
    <>
      <div id="ui-layer">
        <div className="controls">
          <h1>圣诞魔法树</h1>
          
          {!started ? (
             <>
               <p>体验由手势控制的 3D 梦幻圣诞树</p>
               {loading ? (
                 <p style={{color: `#${COLORS.GOLD.toString(16).padStart(6, '0')}`, fontStyle: 'italic'}}>魔法加载中...</p>
               ) : (
                 <button 
                    className={`btn ${userPhotos.length > 0 ? 'highlight' : ''}`} 
                    onClick={handleStart}
                 >
                    {userPhotos.length > 0 ? '照片已添加，开启体验' : '开启体验'}
                 </button>
               )}
             </>
          ) : (
             <>
               <p><strong>手势指南：</strong></p>
               <ul>
                 <li>✊ <strong>握拳：</strong> 聚合圣诞树</li>
                 <li>🖐 <strong>张开：</strong> 漂浮与探索</li>
                 <li>🤏 <strong>捏合：</strong> 抓取照片</li>
               </ul>
               <div id="gesture-feedback">{gesture}</div>
             </>
          )}

          <div style={{marginTop: '20px', borderTop: '1px solid rgba(212,175,55,0.3)', paddingTop: '15px'}}>
             <p style={{marginBottom:'5px'}}>添加你的回忆：</p>
             <div className="file-upload-container">
                 <label className="file-upload-label">
                   {/* Added 'multiple' attribute for batch selection */}
                   <input type="file" ref={fileInputRef} onChange={handleFileUpload} accept="image/*" multiple />
                   + 添加照片 (可多选)
                 </label>
                 {userPhotos.length > 0 && (
                     <button className="clear-btn" onClick={handleClearPhotos}>清空</button>
                 )}
             </div>
             <p style={{fontSize: '0.8rem', color: '#888', marginTop: '5px'}}>
               树上的照片: {userPhotos.length} 
               {userPhotos.length > 0 && !started && " (请点击开启体验)"}
             </p>
          </div>
        </div>
      </div>

      <video 
        ref={videoRef} 
        id="webcam-preview" 
        playsInline 
        muted
        style={{ display: started ? 'block' : 'none' }}
      ></video>

      <div ref={mountRef} id="canvas-container" style={{width: '100%', height: '100%'}} />
    </>
  );
};

const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(<App />);
}
