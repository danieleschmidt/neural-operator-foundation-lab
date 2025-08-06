"""Tests for data augmentation functionality."""

import pytest
import torch
import numpy as np
from unittest.mock import patch, Mock

from neural_operator_lab.data.augmentation import (
    PDEAugmentation, PhysicsAwareAugmentation, 
    CompositeAugmentation, create_augmentation_pipeline,
    mixup_augmentation
)


class TestPDEAugmentation:
    """Test PDE augmentation functionality."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        aug = PDEAugmentation()
        
        assert aug.rotation is True
        assert aug.reflection is True
        assert aug.gaussian_noise == 0.01
        assert aug.initial_condition_perturbation == 0.05
        assert aug.spatial_scaling is True
        assert aug.time_reversal is False
        assert len(aug.augmentations) > 0
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        aug = PDEAugmentation(
            rotation=False,
            gaussian_noise=0.05,
            time_reversal=True,
            random_crop=True,
            crop_size=(16, 16)
        )
        
        assert aug.rotation is False
        assert aug.gaussian_noise == 0.05
        assert aug.time_reversal is True
        assert aug.random_crop is True
        assert aug.crop_size == (16, 16)
    
    def test_augmentation_list_building(self):
        """Test augmentation list building."""
        # All augmentations disabled
        aug = PDEAugmentation(
            rotation=False,
            reflection=False,
            gaussian_noise=0,
            initial_condition_perturbation=0,
            spatial_scaling=False,
            time_reversal=False,
            random_crop=False
        )
        assert len(aug.augmentations) == 0
        
        # Only rotation enabled
        aug = PDEAugmentation(
            rotation=True,
            reflection=False,
            gaussian_noise=0,
            initial_condition_perturbation=0,
            spatial_scaling=False,
            time_reversal=False,
            random_crop=False
        )
        assert len(aug.augmentations) == 1
    
    def test_call_method(self):
        """Test calling augmentation on data."""
        aug = PDEAugmentation(gaussian_noise=0.01)
        data = torch.randn(2, 3, 32, 32)
        
        # Apply augmentation
        augmented = aug(data, probability=1.0)  # Always apply
        
        assert augmented.shape == data.shape
        assert augmented.dtype == data.dtype
        assert torch.isfinite(augmented).all()
    
    def test_probability_control(self):
        """Test augmentation probability control."""
        aug = PDEAugmentation(gaussian_noise=0.01)
        data = torch.randn(2, 3, 32, 32)
        
        # With probability 0, data should be unchanged
        with patch('random.random', return_value=0.9):  # > probability
            augmented = aug(data, probability=0.5)
            torch.testing.assert_close(augmented, data)
        
        # With probability 1, data should be changed (due to noise)
        with patch('random.random', return_value=0.1):  # < probability
            augmented = aug(data, probability=0.5)
            assert not torch.allclose(augmented, data)


class TestRotationAugmentation:
    """Test rotation augmentation."""
    
    def test_random_rotation_2d(self):
        """Test 2D random rotation."""
        aug = PDEAugmentation(
            rotation=True,
            reflection=False,
            gaussian_noise=0,
            initial_condition_perturbation=0,
            spatial_scaling=False
        )
        
        data = torch.randn(1, 3, 16, 16)
        
        with patch('random.choice', return_value=1):  # 90 degree rotation
            rotated = aug._random_rotation(data)
        
        assert rotated.shape == data.shape
        assert torch.isfinite(rotated).all()
    
    def test_no_rotation_1d(self):
        """Test that 1D data is not rotated."""
        aug = PDEAugmentation()
        data = torch.randn(1, 10)  # 1D data
        
        rotated = aug._random_rotation(data)
        torch.testing.assert_close(rotated, data)
    
    def test_rotation_k_values(self):
        """Test different rotation values."""
        aug = PDEAugmentation()
        data = torch.randn(1, 3, 8, 8)
        
        # Test k=0 (no rotation)
        with patch('random.choice', return_value=0):
            rotated = aug._random_rotation(data)
            torch.testing.assert_close(rotated, data)
        
        # Test k=1, 2, 3 (90, 180, 270 degrees)
        for k in [1, 2, 3]:
            with patch('random.choice', return_value=k):
                rotated = aug._random_rotation(data)
                assert rotated.shape == data.shape


class TestReflectionAugmentation:
    """Test reflection augmentation."""
    
    def test_random_reflection(self):
        """Test random reflection."""
        aug = PDEAugmentation()
        data = torch.randn(1, 3, 16, 16)
        
        # Test with deterministic flipping
        with patch('random.random', side_effect=[0.3, 0.3]):  # Both flips
            reflected = aug._random_reflection(data)
        
        assert reflected.shape == data.shape
        assert torch.isfinite(reflected).all()
    
    def test_no_reflection_low_dim(self):
        """Test no reflection for low dimensional data."""
        aug = PDEAugmentation()
        data = torch.randn(1, 10)  # 1D data
        
        reflected = aug._random_reflection(data)
        torch.testing.assert_close(reflected, data)
    
    def test_horizontal_vertical_flips(self):
        """Test horizontal and vertical flips separately."""
        aug = PDEAugmentation()
        data = torch.randn(1, 3, 8, 8)
        
        # Only horizontal flip
        with patch('random.random', side_effect=[0.3, 0.7]):  # First flip, no second
            reflected = aug._random_reflection(data)
            assert reflected.shape == data.shape
        
        # Only vertical flip
        with patch('random.random', side_effect=[0.7, 0.3]):  # No first, second flip
            reflected = aug._random_reflection(data)
            assert reflected.shape == data.shape


class TestNoiseAugmentation:
    """Test noise augmentation."""
    
    def test_gaussian_noise_addition(self):
        """Test Gaussian noise addition."""
        aug = PDEAugmentation(gaussian_noise=0.1)
        data = torch.zeros(2, 3, 8, 8)  # Zero tensor for easy testing
        
        noisy = aug._add_gaussian_noise(data)
        
        assert noisy.shape == data.shape
        assert torch.isfinite(noisy).all()
        assert not torch.allclose(noisy, data)  # Should be different due to noise
        assert torch.abs(noisy).max() > 0  # Should have some non-zero values
    
    def test_noise_scale(self):
        """Test noise scaling."""
        data = torch.zeros(2, 3, 8, 8)
        
        # Small noise
        aug_small = PDEAugmentation(gaussian_noise=0.01)
        noisy_small = aug_small._add_gaussian_noise(data)
        
        # Large noise
        aug_large = PDEAugmentation(gaussian_noise=0.1)
        noisy_large = aug_large._add_gaussian_noise(data)
        
        # Large noise should have higher variance
        assert torch.var(noisy_large) > torch.var(noisy_small)


class TestInitialConditionPerturbation:
    """Test initial condition perturbation."""
    
    def test_perturb_initial_condition(self):
        """Test initial condition perturbation."""
        aug = PDEAugmentation(initial_condition_perturbation=0.1)
        data = torch.randn(2, 10, 3, 16, 16)  # Batch, time, channels, height, width
        
        perturbed = aug._perturb_initial_condition(data)
        
        assert perturbed.shape == data.shape
        assert torch.isfinite(perturbed).all()
        # First time step should be different
        assert not torch.allclose(perturbed[:, 0], data[:, 0])
        # Other time steps should be unchanged
        torch.testing.assert_close(perturbed[:, 1:], data[:, 1:])
    
    def test_no_perturbation_low_dim(self):
        """Test no perturbation for low dimensional data."""
        aug = PDEAugmentation()
        data = torch.randn(2, 3, 16, 16)  # No time dimension
        
        perturbed = aug._perturb_initial_condition(data)
        torch.testing.assert_close(perturbed, data)


class TestSpatialScaling:
    """Test spatial scaling augmentation."""
    
    @patch('torch.nn.functional.interpolate')
    def test_spatial_scaling_2d(self, mock_interpolate):
        """Test 2D spatial scaling."""
        aug = PDEAugmentation()
        data = torch.randn(3, 16, 16)  # 2D data
        
        # Mock interpolation to return scaled data
        mock_interpolate.return_value = torch.randn(1, 3, 20, 20)  # Larger
        
        with patch('random.uniform', return_value=1.25):  # 25% scaling
            scaled = aug._spatial_scaling(data)
        
        assert torch.isfinite(scaled).all()
        mock_interpolate.assert_called()
    
    @patch('torch.nn.functional.interpolate')
    def test_spatial_scaling_4d(self, mock_interpolate):
        """Test 4D spatial scaling (time series)."""
        aug = PDEAugmentation()
        data = torch.randn(10, 3, 16, 16)  # Time, channels, height, width
        
        # Mock interpolation for each time step
        mock_interpolate.return_value = torch.randn(1, 3, 20, 20)
        
        with patch('random.uniform', return_value=1.25):
            scaled = aug._spatial_scaling(data)
        
        assert scaled.shape[0] == 10  # Time dimension preserved
        assert torch.isfinite(scaled).all()
    
    def test_no_scaling_low_dim(self):
        """Test no scaling for low dimensional data."""
        aug = PDEAugmentation()
        data = torch.randn(10)  # 1D data
        
        scaled = aug._spatial_scaling(data)
        torch.testing.assert_close(scaled, data)


class TestTimeReversal:
    """Test time reversal augmentation."""
    
    def test_time_reversal(self):
        """Test time reversal."""
        aug = PDEAugmentation(time_reversal=True)
        data = torch.randn(2, 10, 3, 16, 16)  # Batch, time, channels, height, width
        
        with patch('random.random', return_value=0.3):  # Apply reversal
            reversed_data = aug._time_reversal(data)
        
        assert reversed_data.shape == data.shape
        # Should be different due to time reversal
        assert not torch.allclose(reversed_data, data)
    
    def test_no_reversal_low_dim(self):
        """Test no reversal for low dimensional data."""
        aug = PDEAugmentation(time_reversal=True)
        data = torch.randn(2, 3, 16, 16)  # No time dimension
        
        reversed_data = aug._time_reversal(data)
        torch.testing.assert_close(reversed_data, data)
    
    def test_probabilistic_reversal(self):
        """Test probabilistic time reversal."""
        aug = PDEAugmentation(time_reversal=True)
        data = torch.randn(2, 10, 3, 8, 8)
        
        # Should not apply reversal
        with patch('random.random', return_value=0.7):  # > 0.5
            reversed_data = aug._time_reversal(data)
            torch.testing.assert_close(reversed_data, data)


class TestRandomCrop:
    """Test random cropping augmentation."""
    
    def test_random_crop(self):
        """Test random cropping."""
        aug = PDEAugmentation(random_crop=True, crop_size=(8, 8))
        data = torch.randn(3, 16, 16, 2)  # Channels, height, width, extra_dim
        
        with patch('random.randint', side_effect=[4, 4]):  # Crop position
            cropped = aug._random_crop(data)
        
        assert cropped.shape[-3:-1] == (8, 8)  # Should be cropped size
        assert torch.isfinite(cropped).all()
    
    def test_no_crop_small_data(self):
        """Test no cropping when data is smaller than crop size."""
        aug = PDEAugmentation(random_crop=True, crop_size=(32, 32))
        data = torch.randn(3, 16, 16, 2)  # Smaller than crop size
        
        cropped = aug._random_crop(data)
        torch.testing.assert_close(cropped, data)
    
    def test_no_crop_size_specified(self):
        """Test no cropping when crop size not specified."""
        aug = PDEAugmentation(random_crop=True, crop_size=None)
        data = torch.randn(3, 16, 16)
        
        cropped = aug._random_crop(data)
        torch.testing.assert_close(cropped, data)


class TestPhysicsAwareAugmentation:
    """Test physics-aware augmentation."""
    
    def test_initialization(self):
        """Test physics-aware augmentation initialization."""
        aug = PhysicsAwareAugmentation("navier_stokes")
        
        assert aug.pde_type == "navier_stokes"
        assert aug.time_reversal is False  # NS is not reversible
    
    def test_pde_type_configuration(self):
        """Test PDE-specific configuration."""
        # Navier-Stokes
        aug_ns = PhysicsAwareAugmentation("navier_stokes")
        assert aug_ns.time_reversal is False
        
        # Heat equation
        aug_heat = PhysicsAwareAugmentation("heat")
        assert aug_heat.time_reversal is False
        
        # Wave equation
        aug_wave = PhysicsAwareAugmentation("wave")
        assert aug_wave.time_reversal is True
        
        # Darcy flow
        aug_darcy = PhysicsAwareAugmentation("darcy")
        assert aug_darcy.time_reversal is False
        assert aug_darcy.initial_condition_perturbation == 0
    
    def test_physics_aware_rotation(self):
        """Test physics-aware rotation for Navier-Stokes."""
        aug = PhysicsAwareAugmentation("navier_stokes")
        
        # Data with velocity components (u, v)
        data = torch.randn(1, 1, 8, 8, 2)  # Last dim has u, v components
        
        with patch('random.choice', return_value=1):  # 90 degree rotation
            rotated = aug._random_rotation(data)
        
        assert rotated.shape == data.shape
        assert torch.isfinite(rotated).all()
    
    def test_physics_aware_noise(self):
        """Test physics-aware noise scaling."""
        aug_ns = PhysicsAwareAugmentation("navier_stokes", gaussian_noise=0.1)
        aug_pressure = PhysicsAwareAugmentation("pressure", gaussian_noise=0.1)
        
        data = torch.zeros(2, 3, 8, 8)
        
        # Navier-Stokes should have reduced noise
        noisy_ns = aug_ns._add_gaussian_noise(data)
        
        # Pressure should have even more reduced noise
        noisy_pressure = aug_pressure._add_gaussian_noise(data)
        
        # Check that noise levels are different
        assert torch.var(noisy_ns) != torch.var(noisy_pressure)


class TestCreateAugmentationPipeline:
    """Test augmentation pipeline creation."""
    
    def test_create_physics_aware_pipeline(self):
        """Test creating physics-aware pipeline."""
        config = {
            'physics_aware': True,
            'rotation': True,
            'gaussian_noise': 0.05
        }
        
        pipeline = create_augmentation_pipeline('navier_stokes', config)
        
        assert isinstance(pipeline, PhysicsAwareAugmentation)
        assert pipeline.pde_type == 'navier_stokes'
        assert pipeline.rotation is True
        assert pipeline.gaussian_noise == 0.05
    
    def test_create_standard_pipeline(self):
        """Test creating standard pipeline."""
        config = {
            'physics_aware': False,
            'rotation': True,
            'reflection': False
        }
        
        pipeline = create_augmentation_pipeline('heat', config)
        
        assert isinstance(pipeline, PDEAugmentation)
        assert pipeline.rotation is True
        assert pipeline.reflection is False


class TestCompositeAugmentation:
    """Test composite augmentation."""
    
    def test_initialization(self):
        """Test composite augmentation initialization."""
        aug1 = PDEAugmentation(rotation=True, reflection=False)
        aug2 = PDEAugmentation(gaussian_noise=0.05, rotation=False)
        
        composite = CompositeAugmentation([aug1, aug2])
        
        assert len(composite.augmentations) == 2
        assert composite.augmentations[0] is aug1
        assert composite.augmentations[1] is aug2
    
    def test_sequential_application(self):
        """Test sequential application of augmentations."""
        aug1 = PDEAugmentation(gaussian_noise=0.01, rotation=False, reflection=False,
                              spatial_scaling=False, time_reversal=False, random_crop=False,
                              initial_condition_perturbation=0)
        aug2 = PDEAugmentation(gaussian_noise=0.01, rotation=False, reflection=False,
                              spatial_scaling=False, time_reversal=False, random_crop=False,
                              initial_condition_perturbation=0)
        
        composite = CompositeAugmentation([aug1, aug2])
        data = torch.randn(2, 3, 8, 8)
        
        augmented = composite(data, probability=1.0)
        
        assert augmented.shape == data.shape
        assert torch.isfinite(augmented).all()
        # Should be different due to noise from both augmentations
        assert not torch.allclose(augmented, data)


class TestMixupAugmentation:
    """Test mixup augmentation."""
    
    def test_mixup_basic(self):
        """Test basic mixup functionality."""
        batch_x = torch.randn(4, 3, 8, 8)
        batch_y = torch.randn(4, 3, 8, 8)
        
        mixed_x, mixed_y = mixup_augmentation(batch_x, batch_y, alpha=0.2)
        
        assert mixed_x.shape == batch_x.shape
        assert mixed_y.shape == batch_y.shape
        assert torch.isfinite(mixed_x).all()
        assert torch.isfinite(mixed_y).all()
    
    def test_mixup_alpha_zero(self):
        """Test mixup with alpha=0 (no mixing)."""
        batch_x = torch.randn(4, 3, 8, 8)
        batch_y = torch.randn(4, 3, 8, 8)
        
        mixed_x, mixed_y = mixup_augmentation(batch_x, batch_y, alpha=0.0)
        
        # Should return original data when alpha=0
        torch.testing.assert_close(mixed_x, batch_x)
        torch.testing.assert_close(mixed_y, batch_y)
    
    @patch('numpy.random.beta')
    @patch('torch.randperm')
    def test_mixup_deterministic(self, mock_randperm, mock_beta):
        """Test mixup with deterministic parameters."""
        batch_x = torch.randn(2, 3, 4, 4)
        batch_y = torch.randn(2, 3, 4, 4)
        
        mock_beta.return_value = 0.6
        mock_randperm.return_value = torch.tensor([1, 0])  # Reverse order
        
        mixed_x, mixed_y = mixup_augmentation(batch_x, batch_y, alpha=0.2)
        
        # Check that mixing occurred with lambda=0.6
        expected_x = 0.6 * batch_x + 0.4 * batch_x[[1, 0]]
        expected_y = 0.6 * batch_y + 0.4 * batch_y[[1, 0]]
        
        torch.testing.assert_close(mixed_x, expected_x)
        torch.testing.assert_close(mixed_y, expected_y)
    
    def test_mixup_different_shapes(self):
        """Test mixup with different batch sizes."""
        batch_x = torch.randn(3, 2, 8, 8)
        batch_y = torch.randn(3, 2, 8, 8)
        
        mixed_x, mixed_y = mixup_augmentation(batch_x, batch_y, alpha=0.5)
        
        assert mixed_x.shape == (3, 2, 8, 8)
        assert mixed_y.shape == (3, 2, 8, 8)


class TestCropOrPadUtility:
    """Test crop or pad utility function."""
    
    def test_crop_larger_data(self):
        """Test cropping larger data."""
        aug = PDEAugmentation()
        data = torch.randn(3, 16, 16)
        target_size = (8, 8)
        
        result = aug._crop_or_pad_2d(data, target_size)
        
        assert result.shape[-2:] == target_size
        assert torch.isfinite(result).all()
    
    def test_pad_smaller_data(self):
        """Test padding smaller data."""
        aug = PDEAugmentation()
        data = torch.randn(3, 4, 4)
        target_size = (8, 8)
        
        result = aug._crop_or_pad_2d(data, target_size)
        
        assert result.shape[-2:] == target_size
        assert torch.isfinite(result).all()
    
    def test_same_size_data(self):
        """Test data that's already the correct size."""
        aug = PDEAugmentation()
        data = torch.randn(3, 8, 8)
        target_size = (8, 8)
        
        result = aug._crop_or_pad_2d(data, target_size)
        
        torch.testing.assert_close(result, data)
    
    def test_asymmetric_crop_pad(self):
        """Test asymmetric cropping and padding."""
        aug = PDEAugmentation()
        data = torch.randn(3, 10, 6)  # Need to crop width, pad height
        target_size = (8, 4)
        
        result = aug._crop_or_pad_2d(data, target_size)
        
        assert result.shape[-2:] == target_size
        assert torch.isfinite(result).all()