import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

// --- Configuration & Constants ---
const ORNAMENT_COUNT = 400;
const TREE_HEIGHT = 25;
const TREE_BASE_RADIUS = 10;
const COLORS = {
  GOLD: 0xFFD700,
  RED: 0xB01B2E,
  GREEN: 0x0F4D19,
  WHITE: 0xFFFFFF
};

const SHAPES = ['sphere', 'box'];

// --- Helper Functions ---

// Generate a position on a cone spiral
const getTreePosition = (index: number, total: number) => {
  const y = (index / total) * TREE_HEIGHT - (TREE_HEIGHT / 2); // -Height/2 to Height/2
  const radius = ((TREE_HEIGHT / 2 - y) / TREE_HEIGHT) * TREE_BASE_RADIUS + 0.5;
  const angle = index * 0.5; // Spiral tightness
  const x = Math.cos(angle) * radius;
  const z = Math.sin(angle) * radius;
  return new THREE.Vector3(x, y, z);
};

// Generate a random position in a box volume
const getFloatPosition = () => {
  const range = 30;
  return new THREE.Vector3(
    (Math.random() - 0.5) * range,
    (Math.random() - 0.5) * range,
    (Math.random() - 0.5) * range
  );
};

// --- Components ---

const App = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // App State
  const [loading, setLoading] = useState(true);
  const [started, setStarted] = useState(false);
  const [gesture, setGesture] = useState<string>('Detecting...');
  const [userPhotos, setUserPhotos] = useState<string[]>([]);
  
  // Refs for Three.js & Logic
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const composerRef = useRef<EffectComposer | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const particlesRef = useRef<any[]>([]); // Stores mesh and target data
  const photoMeshesRef = useRef<THREE.Mesh[]>([]);
  const frameIdRef = useRef<number>(0);
  const stateRef = useRef<'TREE' | 'FLOAT' | 'FOCUS'>('TREE');
  const targetCameraPos = useRef(new THREE.Vector3(0, 0, 40));
  const focusedPhotoIndex = useRef<number>(-1);
  const handRotationRef = useRef<{x: number, y: number}>({ x: 0, y: 0 });

  // Initialize MediaPipe
  useEffect(() => {
    const initVision = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
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
    };
    initVision();
  }, []);

  // Initialize Three.js
  useEffect(() => {
    if (!mountRef.current) return;

    // Scene
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x000000, 0.02);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 40);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Post Processing (Bloom)
    const renderScene = new RenderPass(scene, camera);
    const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.5, 0.4, 0.85);
    bloomPass.threshold = 0.2;
    bloomPass.strength = 1.5;
    bloomPass.radius = 0.5;
    
    const composer = new EffectComposer(renderer);
    composer.addPass(renderScene);
    composer.addPass(bloomPass);
    composerRef.current = composer;

    // Lights
    const ambientLight = new THREE.AmbientLight(0x404040, 2);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(COLORS.GOLD, 2, 100);
    pointLight.position.set(10, 10, 10);
    scene.add(pointLight);
    
    const pointLight2 = new THREE.PointLight(COLORS.RED, 2, 100);
    pointLight2.position.set(-10, -10, 10);
    scene.add(pointLight2);
    
    const spotLight = new THREE.SpotLight(0xffffff, 5);
    spotLight.position.set(0, 50, 0);
    spotLight.angle = Math.PI / 6;
    spotLight.penumbra = 1;
    scene.add(spotLight);

    // Initial Particles (Ornaments)
    const geometrySphere = new THREE.SphereGeometry(0.5, 16, 16);
    const geometryBox = new THREE.BoxGeometry(0.8, 0.8, 0.8);
    
    const materialGold = new THREE.MeshStandardMaterial({ 
      color: COLORS.GOLD, metalness: 0.9, roughness: 0.1 
    });
    const materialRed = new THREE.MeshStandardMaterial({ 
      color: COLORS.RED, metalness: 0.6, roughness: 0.3 
    });
    const materialGreen = new THREE.MeshStandardMaterial({ 
      color: COLORS.GREEN, metalness: 0.3, roughness: 0.8 
    });

    const particles: any[] = [];

    for (let i = 0; i < ORNAMENT_COUNT; i++) {
      const isSphere = Math.random() > 0.5;
      const geo = isSphere ? geometrySphere : geometryBox;
      
      let mat;
      const rand = Math.random();
      if (rand < 0.33) mat = materialGold;
      else if (rand < 0.66) mat = materialRed;
      else mat = materialGreen;

      const mesh = new THREE.Mesh(geo, mat);
      
      const treePos = getTreePosition(i, ORNAMENT_COUNT);
      const floatPos = getFloatPosition();

      // Start at tree position
      mesh.position.copy(treePos);
      mesh.userData = {
        treePos: treePos,
        floatPos: floatPos,
        rotationSpeed: new THREE.Vector3(Math.random() * 0.02, Math.random() * 0.02, 0)
      };

      scene.add(mesh);
      particles.push(mesh);
    }
    particlesRef.current = particles;

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
    if (!sceneRef.current || userPhotos.length === 0) return;

    // Add new photos to scene
    const loader = new THREE.TextureLoader();
    
    // We only process the latest added photo to avoid duplicates or re-adding
    // For simplicity, we'll clear and rebuild photo meshes if this list changes (inefficient but safe for small counts)
    // Actually, let's just add the ones that aren't there.
    
    // Simplification: Clear old photo meshes to reset positions
    photoMeshesRef.current.forEach(m => sceneRef.current?.remove(m));
    photoMeshesRef.current = [];
    
    userPhotos.forEach((url, index) => {
      loader.load(url, (texture) => {
        const aspect = texture.image.width / texture.image.height;
        const geo = new THREE.PlaneGeometry(3 * aspect, 3);
        const mat = new THREE.MeshBasicMaterial({ 
          map: texture, 
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.9
        });
        const mesh = new THREE.Mesh(geo, mat);
        
        // Add to particles system logic
        // We inject it into the scene but manage it separately for raycasting
        const totalItems = ORNAMENT_COUNT + userPhotos.length;
        const offsetIndex = ORNAMENT_COUNT + index;
        
        const treePos = getTreePosition(offsetIndex, totalItems);
        // Make photos stick out a bit more on the tree
        treePos.multiplyScalar(1.2); 
        
        const floatPos = getFloatPosition();
        
        mesh.position.copy(treePos);
        mesh.lookAt(0, 0, 0); // Face outward from tree center initially
        
        mesh.userData = {
          treePos: treePos,
          floatPos: floatPos,
          isPhoto: true,
          originalScale: new THREE.Vector3(1, 1, 1)
        };
        
        sceneRef.current?.add(mesh);
        photoMeshesRef.current.push(mesh);
      });
    });

  }, [userPhotos]);

  // Main Loop
  const animate = useCallback(() => {
    if (!started || !cameraRef.current || !sceneRef.current) return;

    // 1. Detect Hand
    let detectedState: 'TREE' | 'FLOAT' | 'FOCUS' | null = null;
    let handPos = { x: 0, y: 0 }; // Normalized -1 to 1

    if (handLandmarkerRef.current && videoRef.current && videoRef.current.currentTime > 0) {
      const results = handLandmarkerRef.current.detectForVideo(videoRef.current, Date.now());
      
      if (results.landmarks && results.landmarks.length > 0) {
        const landmarks = results.landmarks[0]; // 0 is Wrist, 4 Thumb tip, 8 Index tip, 12 Middle, 16 Ring, 20 Pinky
        
        // Logic for gestures
        const wrist = landmarks[0];
        const thumbTip = landmarks[4];
        const indexTip = landmarks[8];
        const middleTip = landmarks[12];
        const ringTip = landmarks[16];
        const pinkyTip = landmarks[20];
        
        // Calculate average distance of tips from wrist
        const tips = [indexTip, middleTip, ringTip, pinkyTip];
        let avgDist = 0;
        tips.forEach(p => {
          const d = Math.sqrt(Math.pow(p.x - wrist.x, 2) + Math.pow(p.y - wrist.y, 2));
          avgDist += d;
        });
        avgDist /= 4;

        // Pinch distance (Thumb to Index)
        const pinchDist = Math.sqrt(Math.pow(thumbTip.x - indexTip.x, 2) + Math.pow(thumbTip.y - indexTip.y, 2));

        // Coordinate for rotation (Use wrist or centroid)
        handPos.x = (wrist.x - 0.5) * 2; // -1 to 1
        handPos.y = (wrist.y - 0.5) * 2; // -1 to 1

        if (pinchDist < 0.05) {
          detectedState = 'FOCUS';
          setGesture('Pinch (Grab)');
        } else if (avgDist < 0.15) {
          detectedState = 'TREE';
          setGesture('Fist (Tree)');
        } else {
          detectedState = 'FLOAT';
          setGesture('Open Hand (Float)');
          // Update rotation ref if in float mode
          handRotationRef.current = { x: handPos.x, y: handPos.y };
        }
      } else {
        setGesture('No Hand Detected');
      }
    }

    // 2. State Transition Logic
    if (detectedState) {
      // Logic: If we are in FOCUS, we only stay in focus if pinch is held OR if we are transitioning.
      // But user requirements: 
      // - Fist -> Tree
      // - Open -> Scatter
      // - Pinch -> Grab Photo
      
      if (detectedState === 'FOCUS') {
         if (stateRef.current !== 'FOCUS') {
             // Try to grab nearest photo
             stateRef.current = 'FOCUS';
             // Raycast logic could go here, but for simplicity, we focus the photo closest to screen center or cycle
             // Let's just pick a random one if none focused, or cycle
             if (photoMeshesRef.current.length > 0) {
               focusedPhotoIndex.current = (focusedPhotoIndex.current + 1) % photoMeshesRef.current.length;
             }
         }
      } else {
        stateRef.current = detectedState;
        focusedPhotoIndex.current = -1;
      }
    }

    // 3. Animation & Interpolation
    const time = Date.now() * 0.001;
    const currentState = stateRef.current;

    // Update Ornaments
    particlesRef.current.forEach((mesh) => {
      let target;
      if (currentState === 'TREE') {
        target = mesh.userData.treePos;
      } else {
        // FLOAT or FOCUS
        target = mesh.userData.floatPos;
      }
      
      // Lerp position
      mesh.position.lerp(target, 0.05);
      
      // Rotate
      mesh.rotation.x += mesh.userData.rotationSpeed.x;
      mesh.rotation.y += mesh.userData.rotationSpeed.y;
    });

    // Update Photos
    photoMeshesRef.current.forEach((mesh, i) => {
      let targetPos = new THREE.Vector3();
      let targetScale = new THREE.Vector3(1, 1, 1);
      let targetRot = new THREE.Quaternion();

      if (currentState === 'TREE') {
        targetPos.copy(mesh.userData.treePos);
        // Look away from center
        const lookAtPos = mesh.position.clone().multiplyScalar(2);
        const m = new THREE.Matrix4();
        m.lookAt(lookAtPos, mesh.position, new THREE.Vector3(0, 1, 0));
        targetRot.setFromRotationMatrix(m);

      } else if (currentState === 'FOCUS' && i === focusedPhotoIndex.current) {
        // Bring to front center
        targetPos.set(0, 0, 15);
        targetScale.set(3, 3, 3);
        targetRot.setFromEuler(new THREE.Euler(0, 0, 0));
      } else {
        // FLOAT or unfocused photos
        targetPos.copy(mesh.userData.floatPos);
        targetRot.setFromEuler(new THREE.Euler(time * 0.2 + i, time * 0.1, 0));
      }

      mesh.position.lerp(targetPos, 0.08);
      mesh.scale.lerp(targetScale, 0.08);
      mesh.quaternion.slerp(targetRot, 0.08);
    });

    // Camera Movement
    if (currentState === 'FLOAT' || currentState === 'FOCUS') {
      // Rotate around based on hand position
      // Map hand x (-1 to 1) to angle
      const angle = handRotationRef.current.x * Math.PI; 
      const height = handRotationRef.current.y * 10;
      
      const r = 40;
      const targetCamX = Math.sin(angle) * r;
      const targetCamZ = Math.cos(angle) * r;
      const targetCamY = -height; // Invert y for natural feel

      cameraRef.current.position.lerp(new THREE.Vector3(targetCamX, targetCamY, targetCamZ), 0.05);
      cameraRef.current.lookAt(0, 0, 0);
    } else {
      // Reset camera for TREE
      cameraRef.current.position.lerp(new THREE.Vector3(0, 0, 40), 0.05);
      cameraRef.current.lookAt(0, 0, 0);
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


  // Handlers
  const handleStart = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.addEventListener('loadeddata', () => {
           setStarted(true);
        });
      }
    } catch (err) {
      console.error("Camera access denied:", err);
      alert("Please allow camera access to use hand gestures.");
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      const url = URL.createObjectURL(file);
      setUserPhotos(prev => [...prev, url]);
    }
  };

  return (
    <>
      {/* UI Overlay */}
      <div id="ui-layer">
        <div className="controls">
          <h1>Christmas Magic</h1>
          
          {!started ? (
             <>
               <p>Experience a gesture-controlled 3D Christmas tree.</p>
               {loading ? (
                 <p style={{color: `#${COLORS.GOLD.toString(16).padStart(6, '0')}`}}>Loading AI Models...</p>
               ) : (
                 <button className="btn" onClick={handleStart}>Start Experience</button>
               )}
             </>
          ) : (
             <>
               <p><strong>Gesture Controls:</strong></p>
               <ul>
                 <li>✊ <strong>Fist:</strong> Assemble Tree</li>
                 <li>🖐 <strong>Open Palm:</strong> Explode / Float</li>
                 <li>🤏 <strong>Pinch:</strong> Grab Photo</li>
                 <li>👋 <strong>Move Hand:</strong> Rotate View (in Float mode)</li>
               </ul>
               <div id="gesture-feedback">{gesture}</div>
             </>
          )}

          <div style={{marginTop: '20px', borderTop: '1px solid #555', paddingTop: '10px'}}>
             <p>Add your memories to the tree:</p>
             <label className="file-upload-label">
               <input type="file" ref={fileInputRef} onChange={handleFileUpload} accept="image/*" />
               + Upload Photo
             </label>
             <p style={{fontSize: '0.8rem', color: '#888'}}>
               Photos Added: {userPhotos.length}
             </p>
          </div>
        </div>
      </div>

      {/* Hidden Video for MediaPipe */}
      <video 
        ref={videoRef} 
        id="webcam-preview" 
        autoPlay 
        playsInline 
        muted
        style={{ display: started ? 'block' : 'none' }}
      ></video>

      {/* Three.js Container */}
      <div ref={mountRef} id="canvas-container" style={{width: '100%', height: '100%'}} />
    </>
  );
};

// Mount
const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(<App />);
}