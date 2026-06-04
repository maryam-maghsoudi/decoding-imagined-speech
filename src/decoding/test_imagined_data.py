"""
Test script to check if imagined MEG data can be loaded correctly.
"""
import sys
sys.path.insert(0, '.')
from contrastive_word_meg_compare import MEGWordDataset, SUBJECTS, POEM_KEYS, ONSET_DIR, REMOVE_FLASHES

def test_imagined_data_loading():
    """Test loading imagined MEG data for one subject."""
    test_subject = "sub-03"
    other_subjects = [s for s in SUBJECTS if s != test_subject]
    
    print(f"Testing imagined MEG data loading...")
    print(f"Test subject: {test_subject}")
    print(f"Training subjects: {other_subjects}")
    
    # Try to load imagined data for training subjects
    print("\nLoading imagined MEG dataset...")
    try:
        img_ds = MEGWordDataset(
            subjects=other_subjects[:2],  # Just test with 2 subjects first
            poem_keys=POEM_KEYS,
            onset_dir=ONSET_DIR,
            cond_suffix="img",  # Imagined data
            remove_flashes=REMOVE_FLASHES,
        )
        
        print(f"Success! Loaded {len(img_ds.pairs)} imagined MEG windows")
        print(f"Vocabulary size: {len(img_ds.vocab)}")
        print(f"Channel count: {img_ds.pairs[0][0].shape[0] if img_ds.pairs else 'N/A'}")
        print(f"First 10 words: {img_ds.words[:10]}")
        
        # Check if we have enough data for training
        if len(img_ds.pairs) < 100:
            print(f"WARNING: Only {len(img_ds.pairs)} samples - may not be enough for training")
        
        return True
        
    except Exception as e:
        print(f"ERROR loading imagined data: {e}")
        return False

def test_listened_data_loading():
    """Test loading listened MEG data for comparison."""
    test_subjects = ["sub-01", "sub-03"]
    
    print(f"\nTesting listened MEG data loading for comparison...")
    try:
        lis_ds = MEGWordDataset(
            subjects=test_subjects,
            poem_keys=POEM_KEYS,
            onset_dir=ONSET_DIR,
            cond_suffix="lis",  # Listened data
            remove_flashes=REMOVE_FLASHES,
        )
        
        print(f"Success! Loaded {len(lis_ds.pairs)} listened MEG windows")
        print(f"Vocabulary size: {len(lis_ds.vocab)}")
        
        return True
        
    except Exception as e:
        print(f"ERROR loading listened data: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("TESTING MEG DATA LOADING")
    print("="*60)
    
    # Test both imagined and listened data loading
    img_success = test_imagined_data_loading()
    lis_success = test_listened_data_loading()
    
    print(f"\nResults:")
    print(f"  Imagined data loading: {'SUCCESS' if img_success else 'FAILED'}")
    print(f"  Listened data loading: {'SUCCESS' if lis_success else 'FAILED'}")
    
    if img_success:
        print(f"\n✓ Ready to proceed with direct imagined MEG training!")
    else:
        print(f"\n✗ Need to fix data loading issues first.")