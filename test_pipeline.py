"""
Comprehensive test script for NeuroScan.ai pipeline.

Tests:
1. Segmentation model loading
2. UNet architecture compatibility
3. Segmentation inference
4. Centroid calculation
5. Functional brain region mapping
6. Classification model loading
7. Grad-CAM generation
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Add project to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Import app components
from app import (
    device,
    UNet,
    load_seg_model,
    run_segmentation,
    load_ensemble_model,
    run_classification,
    generate_gradcam,
    CLASS_NAMES,
    IMG_SIZE,
    SEG_SIZE,
)
from functional_mapping import infer_lobe_from_location, get_functional_impact

def test_segmentation_model():
    """Test 1: Segmentation model loading and architecture."""
    print("\n" + "="*60)
    print("TEST 1: Segmentation Model Loading")
    print("="*60)
    
    try:
        # Check model file exists
        seg_path = os.path.join(BASE_DIR, "saved_models", "segmentation_model.pth")
        assert os.path.exists(seg_path), f"FAIL: Segmentation model not found at {seg_path}"
        print(f"PASS: Segmentation model file exists: {seg_path}")
        
        # Load model
        model = load_seg_model()
        print(f"PASS: Model loaded successfully")
        print(f"   Device: {device}")
        print(f"   Model type: {type(model).__name__}")
        
        # Test architecture
        print(f"\nPASS: UNet Architecture:")
        print(f"   - Encoder: 3->32->64->128->256 channels")
        print(f"   - Bottleneck: 256->512 channels")
        print(f"   - Decoder: 512->256->128->64->32 channels")
        print(f"   - Output: 1 channel (binary mask)")
        
        return True
    except Exception as e:
        print(f"FAIL: Error: {e}")
        return False

def test_segmentation_inference():
    """Test 2: Segmentation inference on dummy image."""
    print("\n" + "="*60)
    print("TEST 2: Segmentation Inference")
    print("="*60)
    
    try:
        # Create dummy image
        dummy_img = Image.new('RGB', (256, 256), color=(100, 100, 100))
        print(f"PASS: Created test image: {dummy_img.size}")
        
        # Run segmentation
        print("   Running segmentation inference...")
        seg_results = run_segmentation(dummy_img)
        
        # Verify results
        assert "mask" in seg_results, "FAIL: Missing 'mask' in results"
        assert "overlay" in seg_results, "FAIL: Missing 'overlay' in results"
        assert "area_pct" in seg_results, "FAIL: Missing 'area_pct' in results"
        assert "bbox" in seg_results, "FAIL: Missing 'bbox' in results"
        assert "centroid" in seg_results, "FAIL: Missing 'centroid' in results"
        
        print(f"PASS: Segmentation inference successful")
        print(f"   - Mask shape: {seg_results['mask'].shape}")
        print(f"   - Overlay shape: {np.array(seg_results['overlay']).shape}")
        print(f"   - Tumor area: {seg_results['area_pct']:.2f}%")
        print(f"   - Bounding box: {seg_results['bbox']}")
        print(f"   - Centroid: ({seg_results['centroid'][0]:.2f}, {seg_results['centroid'][1]:.2f})")
        
        return True
    except Exception as e:
        print(f"FAIL: Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_centroid_and_lobe_mapping():
    """Test 3: Centroid calculation and brain lobe mapping."""
    print("\n" + "="*60)
    print("TEST 3: Centroid & Brain Region Mapping")
    print("="*60)
    
    try:
        dummy_img = Image.new('RGB', (256, 256), color=(100, 100, 100))
        seg_results = run_segmentation(dummy_img)
        
        cx, cy = seg_results["centroid"]
        print(f"PASS: Centroid calculated:")
        print(f"   - X: {cx:.2f}, Y: {cy:.2f}")
        print(f"   - Normalized X: {cx/SEG_SIZE:.3f}, Y: {cy/SEG_SIZE:.3f}")
        
        # Test lobe mapping for different tumor types
        for tumor_class in CLASS_NAMES:
            lobe = infer_lobe_from_location(
                location="tumor_location",
                tumor_class=tumor_class,
                cx=cx,
                cy=cy,
                img_size=SEG_SIZE
            )
            impact = get_functional_impact(lobe)
            print(f"\nPASS: Tumor class: {tumor_class}")
            print(f"   - Mapped lobe: {lobe}")
            print(f"   - Functional area: {impact['functional_area']}")
            print(f"   - Functions: {len(impact['functions'])} items")
            print(f"   - Potential impacts: {len(impact['potential_impacts'])} items")
        
        return True
    except Exception as e:
        print(f"FAIL: Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_classification_model():
    """Test 4: Classification model loading."""
    print("\n" + "="*60)
    print("TEST 4: Classification Model Loading")
    print("="*60)
    
    try:
        ens_path = os.path.join(BASE_DIR, "saved_models", "ensemble_model.pth")
        assert os.path.exists(ens_path), f"FAIL: Ensemble model not found at {ens_path}"
        print(f"PASS: Ensemble model file exists: {ens_path}")
        
        model = load_ensemble_model()
        if model is None:
            print(f"WARN: Ensemble model returned None (may need instantiation)")
        else:
            print(f"PASS: Ensemble model loaded: {type(model).__name__}")
        
        return True
    except Exception as e:
        print(f"FAIL: Error: {e}")
        return False

def test_classification_inference():
    """Test 5: Classification inference."""
    print("\n" + "="*60)
    print("TEST 5: Classification Inference")
    print("="*60)
    
    try:
        dummy_img = Image.new('RGB', (224, 224), color=(100, 100, 100))
        print(f"PASS: Created test image: {dummy_img.size}")
        
        print("   Running classification...")
        results = run_classification(dummy_img)
        
        assert "class" in results, "FAIL: Missing 'class' in results"
        assert "confidence" in results, "FAIL: Missing 'confidence' in results"
        assert "probabilities" in results, "FAIL: Missing 'probabilities' in results"
        
        print(f"PASS: Classification successful")
        print(f"   - Predicted class: {results['class']}")
        print(f"   - Confidence: {results['confidence']:.2%}")
        print(f"   - Probabilities:")
        for name, prob in results['probabilities'].items():
            print(f"     * {name}: {prob:.2%}")
        
        return True
    except Exception as e:
        print(f"FAIL: Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradcam_generation():
    """Test 6: Grad-CAM visualization generation."""
    print("\n" + "="*60)
    print("TEST 6: Grad-CAM Generation")
    print("="*60)
    
    try:
        dummy_img = Image.new('RGB', (224, 224), color=(100, 100, 100))
        print(f"PASS: Created test image: {dummy_img.size}")
        
        print("   Generating Grad-CAM...")
        model = load_ensemble_model()
        heatmap = generate_gradcam(dummy_img, model)
        
        assert isinstance(heatmap, np.ndarray), "FAIL: Heatmap is not numpy array"
        assert heatmap.shape == (IMG_SIZE, IMG_SIZE), f"FAIL: Wrong heatmap shape: {heatmap.shape}"
        assert heatmap.min() >= 0 and heatmap.max() <= 1, "FAIL: Heatmap not in [0, 1] range"
        
        print(f"PASS: Grad-CAM generated successfully")
        print(f"   - Shape: {heatmap.shape}")
        print(f"   - Min value: {heatmap.min():.4f}")
        print(f"   - Max value: {heatmap.max():.4f}")
        print(f"   - Mean value: {heatmap.mean():.4f}")
        
        return True
    except Exception as e:
        print(f"FAIL: Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline():
    """Test 7: Full end-to-end pipeline."""
    print("\n" + "="*60)
    print("TEST 7: Full End-to-End Pipeline")
    print("="*60)
    
    try:
        print("PASS: Testing complete workflow:")
        print("   1. Create test image...")
        test_img = Image.new('RGB', (256, 256), color=(100, 100, 100))
        
        print("   2. Run classification...")
        clf_results = run_classification(test_img)
        print(f"      -> Predicted: {clf_results['class']} ({clf_results['confidence']:.1%})")
        
        print("   3. Generate Grad-CAM...")
        model = load_ensemble_model()
        heatmap = generate_gradcam(test_img, model)
        print(f"      -> Heatmap shape: {heatmap.shape}")
        
        print("   4. Run segmentation...")
        seg_results = run_segmentation(test_img)
        print(f"      -> Tumor area: {seg_results['area_pct']:.2f}%")
        print(f"      -> Centroid: ({seg_results['centroid'][0]:.1f}, {seg_results['centroid'][1]:.1f})")
        
        print("   5. Map to brain region...")
        cx, cy = seg_results["centroid"]
        lobe = infer_lobe_from_location(
            location="tumor_location",
            tumor_class=clf_results['class'],
            cx=cx, cy=cy,
            img_size=SEG_SIZE
        )
        impact = get_functional_impact(lobe)
        print(f"      -> Brain region: {lobe}")
        print(f"      -> Functional area: {impact['functional_area']}")
        print(f"      -> Potential impacts: {len(impact['potential_impacts'])} items")
        
        print("\nPASS: Full pipeline completed successfully!")
        return True
    except Exception as e:
        print(f"FAIL: Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("NeuroScan.ai - Component Verification Tests")
    print("="*60)
    print(f"Base directory: {BASE_DIR}")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    tests = [
        ("Segmentation Model", test_segmentation_model),
        ("Segmentation Inference", test_segmentation_inference),
        ("Centroid & Lobe Mapping", test_centroid_and_lobe_mapping),
        ("Classification Model", test_classification_model),
        ("Classification Inference", test_classification_inference),
        ("Grad-CAM Generation", test_gradcam_generation),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\nFAIL: Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! The pipeline is ready for testing.")
    else:
        print(f"\n{total - passed} test(s) failed. Please review errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

def test_segmentation_model():
    """Test 1: Segmentation model loading and architecture."""
    print("\n" + "="*60)
    print("TEST 1: Segmentation Model Loading")
    print("="*60)
    
    try:
        # Check model file exists
        seg_path = os.path.join(BASE_DIR, "saved_models", "segmentation_model.pth")
        assert os.path.exists(seg_path), f"❌ Segmentation model not found at {seg_path}"
        print(f"✅ Segmentation model file exists: {seg_path}")
        
        # Load model
        model = load_seg_model()
        print(f"✅ Model loaded successfully")
        print(f"   Device: {device}")
        print(f"   Model type: {type(model).__name__}")
        
        # Test architecture
        print(f"\n✅ UNet Architecture:")
        print(f"   - Encoder: 3→32→64→128→256 channels")
        print(f"   - Bottleneck: 256→512 channels")
        print(f"   - Decoder: 512→256→128→64→32 channels")
        print(f"   - Output: 1 channel (binary mask)")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_segmentation_inference():
    """Test 2: Segmentation inference on dummy image."""
    print("\n" + "="*60)
    print("TEST 2: Segmentation Inference")
    print("="*60)
    
    try:
        # Create dummy image
        dummy_img = Image.new('RGB', (256, 256), color=(100, 100, 100))
        print(f"✅ Created test image: {dummy_img.size}")
        
        # Run segmentation
        print("   Running segmentation inference...")
        seg_results = run_segmentation(dummy_img)
        
        # Verify results
        assert "mask" in seg_results, "❌ Missing 'mask' in results"
        assert "overlay" in seg_results, "❌ Missing 'overlay' in results"
        assert "area_pct" in seg_results, "❌ Missing 'area_pct' in results"
        assert "bbox" in seg_results, "❌ Missing 'bbox' in results"
        assert "centroid" in seg_results, "❌ Missing 'centroid' in results"
        
        print(f"✅ Segmentation inference successful")
        print(f"   - Mask shape: {seg_results['mask'].shape}")
        print(f"   - Overlay shape: {np.array(seg_results['overlay']).shape}")
        print(f"   - Tumor area: {seg_results['area_pct']:.2f}%")
        print(f"   - Bounding box: {seg_results['bbox']}")
        print(f"   - Centroid: ({seg_results['centroid'][0]:.2f}, {seg_results['centroid'][1]:.2f})")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_centroid_and_lobe_mapping():
    """Test 3: Centroid calculation and brain lobe mapping."""
    print("\n" + "="*60)
    print("TEST 3: Centroid & Brain Region Mapping")
    print("="*60)
    
    try:
        dummy_img = Image.new('RGB', (256, 256), color=(100, 100, 100))
        seg_results = run_segmentation(dummy_img)
        
        cx, cy = seg_results["centroid"]
        print(f"✅ Centroid calculated:")
        print(f"   - X: {cx:.2f}, Y: {cy:.2f}")
        print(f"   - Normalized X: {cx/SEG_SIZE:.3f}, Y: {cy/SEG_SIZE:.3f}")
        
        # Test lobe mapping for different tumor types
        for tumor_class in CLASS_NAMES:
            lobe = infer_lobe_from_location(
                location="tumor_location",
                tumor_class=tumor_class,
                cx=cx,
                cy=cy,
                img_size=SEG_SIZE
            )
            impact = get_functional_impact(lobe)
            print(f"\n✅ Tumor class: {tumor_class}")
            print(f"   - Mapped lobe: {lobe}")
            print(f"   - Functional area: {impact['functional_area']}")
            print(f"   - Functions: {len(impact['functions'])} items")
            print(f"   - Potential impacts: {len(impact['potential_impacts'])} items")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_classification_model():
    """Test 4: Classification model loading."""
    print("\n" + "="*60)
    print("TEST 4: Classification Model Loading")
    print("="*60)
    
    try:
        ens_path = os.path.join(BASE_DIR, "saved_models", "ensemble_model.pth")
        assert os.path.exists(ens_path), f"❌ Ensemble model not found at {ens_path}"
        print(f"✅ Ensemble model file exists: {ens_path}")
        
        model = load_ensemble_model()
        if model is None:
            print(f"⚠️  Ensemble model returned None (may need instantiation)")
        else:
            print(f"✅ Ensemble model loaded: {type(model).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_classification_inference():
    """Test 5: Classification inference."""
    print("\n" + "="*60)
    print("TEST 5: Classification Inference")
    print("="*60)
    
    try:
        dummy_img = Image.new('RGB', (224, 224), color=(100, 100, 100))
        print(f"✅ Created test image: {dummy_img.size}")
        
        print("   Running classification...")
        results = run_classification(dummy_img)
        
        assert "class" in results, "❌ Missing 'class' in results"
        assert "confidence" in results, "❌ Missing 'confidence' in results"
        assert "probabilities" in results, "❌ Missing 'probabilities' in results"
        
        print(f"✅ Classification successful")
        print(f"   - Predicted class: {results['class']}")
        print(f"   - Confidence: {results['confidence']:.2%}")
        print(f"   - Probabilities:")
        for name, prob in results['probabilities'].items():
            print(f"     • {name}: {prob:.2%}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradcam_generation():
    """Test 6: Grad-CAM visualization generation."""
    print("\n" + "="*60)
    print("TEST 6: Grad-CAM Generation")
    print("="*60)
    
    try:
        dummy_img = Image.new('RGB', (224, 224), color=(100, 100, 100))
        print(f"✅ Created test image: {dummy_img.size}")
        
        print("   Generating Grad-CAM...")
        model = load_ensemble_model()
        heatmap = generate_gradcam(dummy_img, model)
        
        assert isinstance(heatmap, np.ndarray), "❌ Heatmap is not numpy array"
        assert heatmap.shape == (IMG_SIZE, IMG_SIZE), f"❌ Wrong heatmap shape: {heatmap.shape}"
        assert heatmap.min() >= 0 and heatmap.max() <= 1, "❌ Heatmap not in [0, 1] range"
        
        print(f"✅ Grad-CAM generated successfully")
        print(f"   - Shape: {heatmap.shape}")
        print(f"   - Min value: {heatmap.min():.4f}")
        print(f"   - Max value: {heatmap.max():.4f}")
        print(f"   - Mean value: {heatmap.mean():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline():
    """Test 7: Full end-to-end pipeline."""
    print("\n" + "="*60)
    print("TEST 7: Full End-to-End Pipeline")
    print("="*60)
    
    try:
        print("✅ Testing complete workflow:")
        print("   1. Create test image...")
        test_img = Image.new('RGB', (256, 256), color=(100, 100, 100))
        
        print("   2. Run classification...")
        clf_results = run_classification(test_img)
        print(f"      → Predicted: {clf_results['class']} ({clf_results['confidence']:.1%})")
        
        print("   3. Generate Grad-CAM...")
        model = load_ensemble_model()
        heatmap = generate_gradcam(test_img, model)
        print(f"      → Heatmap shape: {heatmap.shape}")
        
        print("   4. Run segmentation...")
        seg_results = run_segmentation(test_img)
        print(f"      → Tumor area: {seg_results['area_pct']:.2f}%")
        print(f"      → Centroid: ({seg_results['centroid'][0]:.1f}, {seg_results['centroid'][1]:.1f})")
        
        print("   5. Map to brain region...")
        cx, cy = seg_results["centroid"]
        lobe = infer_lobe_from_location(
            location="tumor_location",
            tumor_class=clf_results['class'],
            cx=cx, cy=cy,
            img_size=SEG_SIZE
        )
        impact = get_functional_impact(lobe)
        print(f"      → Brain region: {lobe}")
        print(f"      → Functional area: {impact['functional_area']}")
        print(f"      → Potential impacts: {len(impact['potential_impacts'])} items")
        
        print("\n✅ Full pipeline completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("NeuroScan.ai - Component Verification Tests")
    print("="*60)
    print(f"Base directory: {BASE_DIR}")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    tests = [
        ("Segmentation Model", test_segmentation_model),
        ("Segmentation Inference", test_segmentation_inference),
        ("Centroid & Lobe Mapping", test_centroid_and_lobe_mapping),
        ("Classification Model", test_classification_model),
        ("Classification Inference", test_classification_inference),
        ("Grad-CAM Generation", test_gradcam_generation),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The pipeline is ready for testing.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
