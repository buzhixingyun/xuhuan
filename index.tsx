import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import * as THREE from 'three';
// We rely on the importmap to resolve 'three/examples/jsm/...' to the CDN URL
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

// --- Configuration & Constants ---
const ORNAMENT_COUNT = 550; 
const TREE_HEIGHT = 28;
const TREE_BASE_RADIUS = 11;
const COLORS = {
  PINK: 0xFFB7C5,   // Sakura Pink
  GOLD: 0xFFE5B4,   // Peach Gold
  RED: 0xFF8888,    // Soft Red
  WHITE: 0xFFFFF0,  // Ivory
  MINT: 0xA0E6B6    // Soft Mint
};

// --- Helper Functions ---

const getTreePosition = (index: number, total: number) => {
  const normalizedIndex = index / total;
  // Fluffy cone shape
  const y = (1 - normalizedIndex) * TREE_HEIGHT - (TREE_HEIGHT / 2); 
  const radius = Math.pow(normalizedIndex, 0.8) * TREE_BASE_RADIUS;
  const angle = index * 2.39996; // Golden angle
  const x = Math.cos(angle) * radius;
  const z = Math.sin(angle) * radius;
  return new THREE.Vector3(x, y, z);
};

// Orbit params for "Fireflies" effect
const getOrbitParams = () => {
  const radius = 8 + Math.random() * 30; 
  const angle = Math.random() * Math.PI * 2;
  const speed = (0.1 + Math.random() * 0.3) * (Math.random() > 0.5 ? 1 : -1); 
  const y = (Math.random() - 0.5) * 25; 
  return { radius, angle, speed, y };
};

// --- Procedural Texture Generators ---

const createSoftGradientTexture = (colorA: string, colorB: string) => {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) return new THREE.Texture();

  // Radial gradient for a soft, glowing look on spheres
  const grad = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2);
  grad.addColorStop(0, colorA);
  grad.addColorStop(1, colorB);

  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
};

// Create soft particle glow texture
const createParticleTexture = () => {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) return new THREE.Texture();

  const grad = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2);
  grad.addColorStop(0, 'rgba(255, 255, 255, 1)');
  grad.addColorStop(0.3, 'rgba(255, 255, 255, 0.6)');
  grad.addColorStop(1, 'rgba(255, 255, 255, 0)');

  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);

  const texture = new THREE.CanvasTexture(canvas);
  return texture;
};

// Cute Stripes with gloss map baked in logic (visual only)
const createStripeTexture = () => {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) return new THREE.Texture();

  ctx.fillStyle = '#FFF0F5'; // Lavender Blush
  ctx.fillRect(0, 0, size, size);

  ctx.fillStyle = '#FFB7C5'; // Pink
  const stripeCount = 4;
  const stripeWidth = size / stripeCount;
  
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

// --- Data Structures for InstancedMesh ---

type InstanceData = {
  treePos: THREE.Vector3;
  // Orbit Data
  orbitRadius: number;
  orbitStartAngle: number;
  orbitSpeed: number;
  orbitY: number;
  
  // Instance info
  meshIndex: number;
  instanceId: number;
  currentPos: THREE.Vector3;
  currentRot: THREE.Euler;
  currentScale: number;
  rotationSpeed: THREE.Vector3;
  phase: number;
};

const App = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cursorRef = useRef<HTMLDivElement>(null);
  
  const [loading, setLoading] = useState(true);
  const [started, setStarted] = useState(false);
  const [gesture, setGesture] = useState<string>('等待唤醒...');
  const [userPhotos, setUserPhotos] = useState<string[]>([]);
  
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const composerRef = useRef<EffectComposer | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  
  const instancedMeshesRef = useRef<THREE.InstancedMesh[]>([]);
  const particlesDataRef = useRef<InstanceData[]>([]);
  const photoMeshesRef = useRef<THREE.Mesh[]>([]);
  const starRef = useRef<THREE.Group | null>(null);
  const stardustRef = useRef<THREE.Points | null>(null);
  const frameIdRef = useRef<number>(0);

  const ambientLightRef = useRef<THREE.AmbientLight | null>(null);
  const mainLightRef = useRef<THREE.PointLight | null>(null);
  const fillLightRef = useRef<THREE.PointLight | null>(null);
  
  const stateRef = useRef<'TREE' | 'FLOAT' | 'FOCUS'>('TREE');
  const focusedPhotoIndex = useRef<number>(-1);
  
  // Tracking & Smooth Refs
  const targetHandPosRef = useRef<{x: number, y: number}>({ x: 0, y: 0 });
  const visualHandPosRef = useRef<{x: number, y: number}>({ x: 0, y: 0 });
  const lastDetectionTimeRef = useRef<number>(0);
  
  const panRef = useRef<{x: number, y: number}>({ x: 0, y: 0 });
  const cameraOffsetRef = useRef<{x: number, y: number}>({ x: 0, y: 0 });
  const zoomRef = useRef<number>(1.0);
  const isDraggingRef = useRef<boolean>(false);
  const lastMousePosRef = useRef<{x: number, y: number}>({ x: 0, y: 0 });
  
  const gestureHistoryRef = useRef<string[]>([]);
  const GESTURE_HISTORY_LIMIT = 5; 
  const lastGestureTimeRef = useRef<number>(0);

  // Init MediaPipe
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
      }
    };
    initVision();
  }, []);

  // Init Three.js
  useEffect(() => {
    if (!mountRef.current) return;

    // Soft Romantic Scene
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x2e1c2b, 0.015); // Lighter fog for dreamy look
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 48);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Post Processing - Dreamy Bloom
    const renderScene = new RenderPass(scene, camera);
    const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.5, 0.4, 0.85);
    // Tuned for soft romantic glow
    bloomPass.threshold = 0.75; // Only glow highlights
    bloomPass.strength = 0.8;   // Distinct glow
    bloomPass.radius = 1.1;     // Wide, soft spread
    
    const composer = new EffectComposer(renderer);
    composer.addPass(renderScene);
    composer.addPass(bloomPass);
    composerRef.current = composer;

    // Lights - Warm & Cozy
    // Ambient: Lavender Blush, Soft
    const ambientLight = new THREE.AmbientLight(0xfff0f5, 0.6); 
    scene.add(ambientLight);
    ambientLightRef.current = ambientLight;

    // Main: Pastel Pink Gold, Bright
    const mainLight = new THREE.PointLight(0xffd1dc, 1.5, 120); 
    mainLight.position.set(10, 20, 20);
    scene.add(mainLight);
    mainLightRef.current = mainLight;
    
    // Fill: Light Cyan, Cooler rim light
    const fillLight = new THREE.PointLight(0xe0ffff, 0.8, 100); 
    fillLight.position.set(-15, -10, 15);
    scene.add(fillLight);
    fillLightRef.current = fillLight;

    // --- Cute Materials & Geometry (Upgraded) ---
    const stripeTexture = createStripeTexture();
    const pinkGradTexture = createSoftGradientTexture('#FFB7C5', '#FFF0F5');
    const goldGradTexture = createSoftGradientTexture('#FFE5B4', '#FFF8DC');

    // 1. Pearl (Iridescent White)
    const matPearl = new THREE.MeshPhysicalMaterial({ 
      color: COLORS.WHITE, 
      metalness: 0.1, 
      roughness: 0.2, 
      clearcoat: 0.9,
      clearcoatRoughness: 0.1,
      sheen: 1.0,
      sheenColor: 0xffe7e7, // Pinkish sheen
      iridescence: 0.6,
      iridescenceIOR: 1.3
    });
    
    // 2. Jelly Pink (Translucent Gummy)
    const matJellyPink = new THREE.MeshPhysicalMaterial({ 
      map: pinkGradTexture,
      color: 0xffffff,
      roughness: 0.15,
      metalness: 0.1,
      transmission: 0.6, // More translucent
      thickness: 2.0,
      clearcoat: 1.0,
      side: THREE.DoubleSide
    });
    
    // 3. Soft Gold (Satin Finish)
    const matGold = new THREE.MeshPhysicalMaterial({ 
      map: goldGradTexture,
      color: 0xffffff,
      roughness: 0.3,
      metalness: 0.7,
      clearcoat: 0.6,
      sheen: 1.0,
      sheenColor: 0xffd700
    });
    
    // 4. Mint Glass (Frosted)
    const matMint = new THREE.MeshPhysicalMaterial({ 
      color: COLORS.MINT,
      roughness: 0.35,
      metalness: 0.1,
      transmission: 0.3,
      clearcoat: 1.0
    });
    
    // 5. Hard Candy (Striped Glossy)
    const matCandy = new THREE.MeshPhysicalMaterial({
        map: stripeTexture, 
        roughness: 0.2,
        metalness: 0.0,
        clearcoat: 1.0, 
        clearcoatRoughness: 0.05
    });

    // --- Geometries: Round and Soft ---
    const geoSphere = new THREE.SphereGeometry(0.6, 32, 32); 
    const geoBox = new THREE.BoxGeometry(0.85, 0.85, 0.85); 
    const geoTorus = new THREE.TorusGeometry(0.4, 0.22, 16, 32);

    const meshDefinitions = [
        { id: 'jelly_pink', geo: geoSphere, mat: matJellyPink },
        { id: 'gold_satin', geo: geoSphere, mat: matGold },
        { id: 'pearl', geo: geoSphere, mat: matPearl },
        { id: 'mint_ring', geo: geoTorus, mat: matMint },
        { id: 'candy_box', geo: geoBox, mat: matCandy },
    ];

    const distribution = [0.3, 0.25, 0.2, 0.15, 0.1]; 
    const counts = distribution.map(d => Math.floor(ORNAMENT_COUNT * d));
    
    const instancedMeshes: THREE.InstancedMesh[] = [];
    const particlesData: InstanceData[] = [];
    
    meshDefinitions.forEach((def, index) => {
        const count = counts[index];
        const mesh = new THREE.InstancedMesh(def.geo, def.mat, count);
        mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        scene.add(mesh);
        instancedMeshes.push(mesh);

        const dummy = new THREE.Object3D();
        for(let i=0; i<count; i++) {
            const totalIndex = particlesData.length;
            const treePos = getTreePosition(totalIndex, ORNAMENT_COUNT);
            const orbitParams = getOrbitParams();
            
            dummy.position.copy(treePos);
            dummy.rotation.set(Math.random()*Math.PI, Math.random()*Math.PI, 0);
            const scale = 0.5 + Math.random() * 0.7;
            dummy.scale.setScalar(scale);
            dummy.updateMatrix();
            
            mesh.setMatrixAt(i, dummy.matrix);

            particlesData.push({
                treePos,
                orbitRadius: orbitParams.radius,
                orbitStartAngle: orbitParams.angle,
                orbitSpeed: orbitParams.speed,
                orbitY: orbitParams.y,
                meshIndex: index,
                instanceId: i,
                currentPos: treePos.clone(),
                currentRot: new THREE.Euler(dummy.rotation.x, dummy.rotation.y, dummy.rotation.z),
                currentScale: scale,
                rotationSpeed: new THREE.Vector3(
                    (Math.random()-0.5)*0.02, (Math.random()-0.5)*0.02, (Math.random()-0.5)*0.02
                ),
                phase: Math.random() * Math.PI * 2,
            });
        }
    });

    instancedMeshesRef.current = instancedMeshes;
    particlesDataRef.current = particlesData;

    // --- Stardust System ---
    const stardustCount = 1500;
    const stardustGeo = new THREE.BufferGeometry();
    const stardustPos = new Float32Array(stardustCount * 3);
    
    for(let i=0; i<stardustCount; i++) {
        const r = 15 + Math.random() * 45;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        
        stardustPos[i*3] = r * Math.sin(phi) * Math.cos(theta);
        stardustPos[i*3+1] = (Math.random() - 0.5) * 60; 
        stardustPos[i*3+2] = r * Math.sin(phi) * Math.sin(theta);
    }
    stardustGeo.setAttribute('position', new THREE.BufferAttribute(stardustPos, 3));
    
    const stardustMat = new THREE.PointsMaterial({
        color: 0xFFF0F5,
        size: 0.5, // Slightly larger
        map: createParticleTexture(),
        transparent: true,
        opacity: 0.5,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
    });
    
    const stardust = new THREE.Points(stardustGeo, stardustMat);
    scene.add(stardust);
    stardustRef.current = stardust;


    // --- Cute Star (Glowing Heart-ish Shape) ---
    const starGroup = new THREE.Group();
    const starGeo = new THREE.OctahedronGeometry(1.5, 0); 
    const starMat = new THREE.MeshBasicMaterial({ color: 0xfffeb8, transparent: true, opacity: 0.9 });
    const starMesh = new THREE.Mesh(starGeo, starMat);
    
    // Halo
    const haloGeo = new THREE.SphereGeometry(2.5, 16, 16);
    const haloMat = new THREE.MeshBasicMaterial({ color: 0xffd700, transparent: true, opacity: 0.2 });
    const haloMesh = new THREE.Mesh(haloGeo, haloMat);

    starGroup.add(starMesh);
    starGroup.add(haloMesh);
    starGroup.position.set(0, TREE_HEIGHT/2 + 2, 0);
    scene.add(starGroup);
    starRef.current = starGroup;

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
  
  // --- Input & Interaction (Keep logic, just sync) ---
  useEffect(() => {
    const handleMouseDown = (e: MouseEvent | TouchEvent) => {
       if (stateRef.current !== 'FOCUS' && stateRef.current !== 'FLOAT') return;
       isDraggingRef.current = true;
       const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
       const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
       lastMousePosRef.current = { x: clientX, y: clientY };
       lastDetectionTimeRef.current = Date.now(); // Keep active
    };
    
    const handleMouseMove = (e: MouseEvent | TouchEvent) => {
        if (!isDraggingRef.current) return;
        lastDetectionTimeRef.current = Date.now();

        const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
        const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
        const deltaX = clientX - lastMousePosRef.current.x;
        const deltaY = clientY - lastMousePosRef.current.y;
        lastMousePosRef.current = { x: clientX, y: clientY };

        if (stateRef.current === 'FOCUS') {
            const sensitivity = 0.03 / zoomRef.current;
            panRef.current.x += deltaX * sensitivity;
            panRef.current.y -= deltaY * sensitivity;
        } else if (stateRef.current === 'FLOAT') {
            const sensitivity = 0.002;
            cameraOffsetRef.current.x -= deltaX * sensitivity;
            cameraOffsetRef.current.y += deltaY * sensitivity;
        }
    };
    
    const handleMouseUp = () => { isDraggingRef.current = false; };
    const handleWheel = (e: WheelEvent) => {
        if (stateRef.current !== 'FOCUS' && stateRef.current !== 'FLOAT') return;
        zoomRef.current += e.deltaY * -0.001;
        zoomRef.current = Math.max(0.5, Math.min(zoomRef.current, 3.0));
        lastDetectionTimeRef.current = Date.now();
    };

    window.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    window.addEventListener('touchstart', handleMouseDown);
    window.addEventListener('touchmove', handleMouseMove);
    window.addEventListener('touchend', handleMouseUp);
    window.addEventListener('wheel', handleWheel);

    return () => {
        window.removeEventListener('mousedown', handleMouseDown);
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
        window.removeEventListener('touchstart', handleMouseDown);
        window.removeEventListener('touchmove', handleMouseMove);
        window.removeEventListener('touchend', handleMouseUp);
        window.removeEventListener('wheel', handleWheel);
    };
  }, []);

  // Photo Uploads
  useEffect(() => {
    if (!sceneRef.current) return;
    const loader = new THREE.TextureLoader();
    photoMeshesRef.current.forEach(m => sceneRef.current?.remove(m));
    photoMeshesRef.current = [];
    
    userPhotos.forEach((url, index) => {
      loader.load(url, (texture) => {
        texture.colorSpace = THREE.SRGBColorSpace;
        const aspect = texture.image.width / texture.image.height;
        const geo = new THREE.PlaneGeometry(4 * aspect, 4);
        const mat = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
        const mesh = new THREE.Mesh(geo, mat);
        
        // Cute Frame
        const frameGeo = new THREE.PlaneGeometry(4 * aspect + 0.3, 4 + 0.3);
        const frameMat = new THREE.MeshPhysicalMaterial({ 
            color: COLORS.WHITE, 
            roughness: 0.5, 
            emissive: COLORS.PINK, 
            emissiveIntensity: 0.2,
            clearcoat: 0.5
        });
        const frameMesh = new THREE.Mesh(frameGeo, frameMat);
        frameMesh.position.z = -0.05;
        
        const group = new THREE.Group();
        group.add(mesh);
        group.add(frameMesh);
        
        const totalItems = userPhotos.length;
        const y = (index / totalItems) * (TREE_HEIGHT * 0.8) - (TREE_HEIGHT * 0.4);
        const angle = index * (Math.PI * 2 / 1.618);
        const radius = TREE_BASE_RADIUS * 0.9;
        const treePos = new THREE.Vector3(Math.cos(angle) * radius, y, Math.sin(angle) * radius).multiplyScalar(1.2);
        
        const orbitParams = getOrbitParams();
        
        group.position.copy(treePos);
        group.lookAt(0, 0, 0);
        
        group.userData = {
          treePos: treePos,
          orbitRadius: orbitParams.radius,
          orbitSpeed: orbitParams.speed,
          orbitY: orbitParams.y,
          orbitAngle: orbitParams.angle,
          isPhoto: true
        };
        
        sceneRef.current?.add(group);
        photoMeshesRef.current.push(group as unknown as THREE.Mesh);
      });
    });
  }, [userPhotos]);

  // Main Animation Loop
  const animate = useCallback(() => {
    if (!started || !cameraRef.current || !sceneRef.current) return;

    const now = Date.now();
    const time = now * 0.001;

    // --- AI Logic (Throttled) ---
    let currentFrameGesture: 'TREE' | 'FLOAT' | 'FOCUS' | null = null;
    if (now - lastGestureTimeRef.current > 40) {
        lastGestureTimeRef.current = now;
        if (handLandmarkerRef.current && videoRef.current && videoRef.current.currentTime > 0) {
          try {
            const results = handLandmarkerRef.current.detectForVideo(videoRef.current, now);
            if (results.landmarks && results.landmarks.length > 0) {
              lastDetectionTimeRef.current = now;
              const landmarks = results.landmarks[0];
              const wrist = landmarks[0];
              const thumbTip = landmarks[4];
              const indexTip = landmarks[8];
              
              const rawX = (wrist.x - 0.5) * 2;
              const rawY = (wrist.y - 0.5) * 2;
              targetHandPosRef.current = { x: rawX, y: rawY };

              const fingerTips = [8, 12, 16, 20];
              const fingerPIPs = [6, 10, 14, 18];
              let openCount = 0;
              for(let i=0; i<4; i++) {
                  const tip = landmarks[fingerTips[i]];
                  const pip = landmarks[fingerPIPs[i]];
                  if (Math.hypot(tip.x - wrist.x, tip.y - wrist.y) > Math.hypot(pip.x - wrist.x, pip.y - wrist.y) * 1.05) {
                      openCount++;
                  }
              }
              const indexMCP = landmarks[5];
              if (Math.hypot(thumbTip.x - indexMCP.x, thumbTip.y - indexMCP.y) > 0.1) openCount++;

              const pinchDist = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);

              if (pinchDist < 0.05) currentFrameGesture = 'FOCUS';
              else if (openCount <= 1) currentFrameGesture = 'TREE';
              else if (openCount >= 3) currentFrameGesture = 'FLOAT';
            }
          } catch (err) {}
        }
    }

    // Auto Reset
    if (now - lastDetectionTimeRef.current > 2000) {
        if (stateRef.current !== 'TREE') {
            stateRef.current = 'TREE';
            setGesture('自动合拢 (休息中)');
            gestureHistoryRef.current = [];
        }
    } else {
        if (currentFrameGesture) gestureHistoryRef.current.push(currentFrameGesture);
        if (gestureHistoryRef.current.length > GESTURE_HISTORY_LIMIT) gestureHistoryRef.current.shift();

        const counts = { TREE: 0, FLOAT: 0, FOCUS: 0 };
        gestureHistoryRef.current.forEach(g => { if (g in counts) counts[g as keyof typeof counts]++; });
        let detectedState: 'TREE' | 'FLOAT' | 'FOCUS' | null = null;
        if (counts.FOCUS >= 3) detectedState = 'FOCUS';
        else if (counts.TREE >= 3) detectedState = 'TREE';
        else if (counts.FLOAT >= 3) detectedState = 'FLOAT';

        if (detectedState) {
            setGesture(detectedState === 'TREE' ? '✊ 收集美好' : detectedState === 'FLOAT' ? '🖐 释放浪漫' : '🤏 珍藏回忆');
            if (detectedState === 'FOCUS' && stateRef.current !== 'FOCUS' && photoMeshesRef.current.length > 0) {
                stateRef.current = 'FOCUS';
                focusedPhotoIndex.current = (focusedPhotoIndex.current + 1) % photoMeshesRef.current.length;
                panRef.current = { x: 0, y: 0 };
                zoomRef.current = 1.0;
            } else if (detectedState !== 'FOCUS') {
                if (stateRef.current !== detectedState) {
                     if (detectedState === 'FLOAT') {
                        cameraOffsetRef.current = { x: 0, y: 0 };
                        zoomRef.current = 1.0;
                     }
                }
                stateRef.current = detectedState;
                focusedPhotoIndex.current = -1;
            }
        }
    }

    // --- Visual Updates ---

    // Hand Cursor Smoothing
    visualHandPosRef.current.x += (targetHandPosRef.current.x - visualHandPosRef.current.x) * 0.15;
    visualHandPosRef.current.y += (targetHandPosRef.current.y - visualHandPosRef.current.y) * 0.15;

    if (cursorRef.current) {
        const cx = (visualHandPosRef.current.x + 1) / 2 * window.innerWidth;
        const cy = (visualHandPosRef.current.y + 1) / 2 * window.innerHeight;
        cursorRef.current.style.transform = `translate(${window.innerWidth - cx}px, ${cy}px)`;
        cursorRef.current.style.opacity = (now - lastDetectionTimeRef.current < 500) ? '1' : '0';
    }

    // Lights Breathing (Cozy)
    if (mainLightRef.current) mainLightRef.current.intensity = 1.5 + Math.sin(time * 1.5) * 0.4;
    if (fillLightRef.current) fillLightRef.current.intensity = 0.8 + Math.cos(time * 1.2) * 0.2;

    // Star Animation
    if (starRef.current) {
        starRef.current.children[0].rotation.y = time * 0.5;
        starRef.current.children[1].scale.setScalar(1.0 + Math.sin(time * 3) * 0.1); 
        
        if (stateRef.current === 'TREE') starRef.current.position.lerp(new THREE.Vector3(0, TREE_HEIGHT/2 + 2, 0), 0.05);
        else starRef.current.position.lerp(new THREE.Vector3(0, 8, 0), 0.05);
    }
    
    // Stardust Animation
    if (stardustRef.current) {
        stardustRef.current.rotation.y = time * 0.03;
        stardustRef.current.position.y = Math.sin(time * 0.2) * 1.5;
        // Twinkle opacity
        (stardustRef.current.material as THREE.PointsMaterial).opacity = 0.5 + Math.sin(time * 3) * 0.15;
    }

    // --- Particles Animation (Soft & Dreamy) ---
    const dummy = new THREE.Object3D();

    particlesDataRef.current.forEach((p) => {
        let target = new THREE.Vector3();

        if (stateRef.current === 'TREE') {
            target.copy(p.treePos);
            // Gentle Twinkle
            p.currentScale = 1.0 + Math.sin(time * 5 + p.phase) * 0.1;
        } else {
            // Floating Fireflies/Snowfall
            const angle = p.orbitStartAngle + time * p.orbitSpeed * 0.5;
            const r = p.orbitRadius + Math.sin(time + p.phase) * 2;
            target.set(
                Math.cos(angle) * r,
                p.orbitY + Math.sin(time * 0.8 + p.phase) * 3, // Vertical drift
                Math.sin(angle) * r
            );
            p.currentScale = 0.8 + Math.sin(time * 2 + p.phase) * 0.2;
        }

        p.currentPos.lerp(target, 0.05);
        p.currentRot.x += p.rotationSpeed.x;
        p.currentRot.y += p.rotationSpeed.y;
        
        dummy.position.copy(p.currentPos);
        dummy.rotation.set(p.currentRot.x, p.currentRot.y, p.currentRot.z);
        dummy.scale.setScalar(p.currentScale);
        dummy.updateMatrix();
        
        if (instancedMeshesRef.current[p.meshIndex]) {
            instancedMeshesRef.current[p.meshIndex].setMatrixAt(p.instanceId, dummy.matrix);
        }
    });

    instancedMeshesRef.current.forEach(mesh => mesh.instanceMatrix.needsUpdate = true);

    // Photos Animation
    photoMeshesRef.current.forEach((group, i) => {
      const mesh = group as unknown as THREE.Group;
      const data = group.userData;
      let targetPos = new THREE.Vector3();
      let targetScale = new THREE.Vector3(1, 1, 1);
      let targetRot = new THREE.Quaternion();

      if (stateRef.current === 'TREE') {
        targetPos.copy(data.treePos);
        const m = new THREE.Matrix4();
        m.lookAt(targetPos.clone().multiplyScalar(2), targetPos, new THREE.Vector3(0, 1, 0));
        targetRot.setFromRotationMatrix(m);
        targetScale.setScalar(0.8);
      } else if (stateRef.current === 'FOCUS' && i === focusedPhotoIndex.current) {
        targetPos.set(panRef.current.x, panRef.current.y, 25);
        targetScale.setScalar(3.5 * zoomRef.current);
        targetRot.setFromEuler(new THREE.Euler(0, 0, 0));
      } else {
        const angle = data.orbitAngle + time * data.orbitSpeed * 0.3;
        targetPos.set(
            Math.cos(angle) * data.orbitRadius,
            data.orbitY + Math.sin(time + i) * 2,
            Math.sin(angle) * data.orbitRadius
        );
        const m = new THREE.Matrix4();
        m.lookAt(cameraRef.current.position, targetPos, new THREE.Vector3(0,1,0));
        targetRot.setFromRotationMatrix(m);
      }
      mesh.position.lerp(targetPos, 0.05);
      mesh.scale.lerp(targetScale, 0.05);
      mesh.quaternion.slerp(targetRot, 0.05);
    });

    // Camera
    if (stateRef.current === 'FLOAT') {
      const horizontalAngle = (visualHandPosRef.current.x + cameraOffsetRef.current.x) * Math.PI * 1.5;
      const elevationY = (visualHandPosRef.current.y + cameraOffsetRef.current.y) * 30;
      const radius = 65 / Math.max(0.5, zoomRef.current);
      
      const targetCamPos = new THREE.Vector3(
          Math.sin(horizontalAngle) * radius,
          -elevationY, 
          Math.cos(horizontalAngle) * radius
      );
      cameraRef.current.position.lerp(targetCamPos, 0.04);
      cameraRef.current.lookAt(0, 0, 0);
    } else if (stateRef.current === 'FOCUS') {
      cameraRef.current.position.lerp(new THREE.Vector3(0, 0, 45), 0.05);
      cameraRef.current.lookAt(0, 0, 0);
    } else {
      cameraRef.current.position.lerp(new THREE.Vector3(0, 0, 52), 0.04);
      cameraRef.current.lookAt(0, 6, 0);
    }

    composerRef.current?.render();
    frameIdRef.current = requestAnimationFrame(animate);
  }, [started]);

  useEffect(() => {
    if (started) frameIdRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frameIdRef.current);
  }, [started, animate]);

  const handleStart = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert("请使用 HTTPS 或 localhost 访问以启用摄像头。");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadeddata = () => {
           videoRef.current?.play().catch(console.error);
           setStarted(true);
        };
      }
    } catch (err: any) {
      alert("摄像头启动失败: " + err.message);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newPhotos: string[] = [];
      Array.from(e.target.files).forEach(file => newPhotos.push(URL.createObjectURL(file)));
      setUserPhotos(prev => [...prev, ...newPhotos]);
    }
  };
  
  return (
    <>
      <div id="ui-layer">
        {!started ? (
            // Landing Page State
            <div className="landing-overlay">
                <div className="landing-card">
                    <h1>Romantic Christmas</h1>
                    <p className="subtitle">✨ 唤醒指尖的浪漫魔法 ✨</p>
                    
                    <div className="file-upload-container">
                        <label className="file-upload-label">
                            <input type="file" ref={fileInputRef} onChange={handleFileUpload} accept="image/*" multiple />
                            📷 添加珍贵的回忆照片 (可选)
                        </label>
                        {userPhotos.length > 0 && <span className="photo-count">已选择 {userPhotos.length} 张美好回忆</span>}
                    </div>

                    {loading ? (
                        <p style={{color:'#ffb7c5', marginTop:'20px'}}>正在准备魔法 (加载资源中)...</p>
                    ) : (
                        <button className={`btn ${userPhotos.length > 0 ? 'highlight' : ''}`} onClick={handleStart}>
                            {userPhotos.length > 0 ? '✨ 带着回忆开启魔法' : '💫 开启梦幻体验'}
                        </button>
                    )}
                </div>
            </div>
        ) : (
            // HUD State (Side Panel)
            <div className="hud-overlay">
                <div className="controls-hud">
                    <h1>Romantic Xmas</h1>
                    <p><strong>魔法手势指南:</strong></p>
                    <ul>
                        <li>✊ <strong>握拳</strong> : 收集美好</li>
                        <li>🖐 <strong>张开</strong> : 释放浪漫</li>
                        <li>🤏 <strong>捏合</strong> : 珍藏回忆</li>
                    </ul>
                    <div id="gesture-feedback">{gesture}</div>
                </div>
            </div>
        )}
      </div>
      <div ref={cursorRef} id="hand-cursor"></div>
      <video ref={videoRef} id="webcam-preview" playsInline muted style={{ display: started ? 'block' : 'none' }}></video>
      <div ref={mountRef} id="canvas-container" style={{width: '100%', height: '100%'}} />
    </>
  );
};

const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(<App />);
}
