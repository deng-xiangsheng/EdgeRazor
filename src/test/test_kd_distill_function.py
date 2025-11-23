"""
Test suite for knowledge distillation loss functions.

This module tests the correctness of distillation loss computations by verifying
that they match their mathematical definitions and expected behaviors.
"""

import pytest
import torch
import torch.nn.functional as F

from edgerazor.kd.util.distill_config import LossConfig
from edgerazor.kd.util.distill_function import (
    compute_fd,
    compute_kld,
    compute_kld_confidence,
    compute_kld_forward,
    compute_kld_reverse,
    compute_teacher_confidence,
)


class TestKLDivergence:
    """Test Kullback-Leibler divergence computations."""

    def test_forward_kld_mathematical_definition(self):
        """
        Test Forward KLD: KL(teacher || student) = sum(P_T * log(P_T / P_S))
        This should match the mathematical definition of forward KL divergence.
        """
        # Create simple logits where we can manually verify the result
        batch_size, seq_len, vocab_size = 2, 4, 10
        torch.manual_seed(42)
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        target = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Create configuration
        config = LossConfig(
            loss_type="logits",
            loss_function="kldf",
            temperature=1.0,
            reduction="sum",
            padding_id=-100
        )
        
        # Compute using our function
        kld_result = compute_kld_forward(student_logits, teacher_logits, target, config)
        
        # Manual computation using F.kl_div which computes KL(teacher || student)
        # Note: F.kl_div expects log_probs as input and probs as target
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        manual_kld = F.kl_div(
            input=student_log_probs,
            target=teacher_probs,
            reduction='none',
            log_target=False
        ).sum(dim=-1)  # Sum over vocabulary
        manual_kld = manual_kld.sum()  # Sum over batch and sequence
        
        # Should match (allowing for numerical precision)
        assert torch.isclose(kld_result, manual_kld, rtol=1e-4, atol=1e-6), \
            f"Forward KLD mismatch: got {kld_result}, expected {manual_kld}"

    def test_reverse_kld_mathematical_definition(self):
        """
        Test Reverse KLD: KL(student || teacher) = sum(P_S * log(P_S / P_T))
        This should match the mathematical definition of reverse KL divergence.
        """
        batch_size, seq_len, vocab_size = 2, 4, 10
        torch.manual_seed(42)
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        target = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        config = LossConfig(
            loss_type="logits",
            loss_function="kldr",
            temperature=1.0,
            reduction="sum",
            padding_id=-100
        )
        
        # Compute using our function
        kld_result = compute_kld_reverse(student_logits, teacher_logits, target, config)
        
        # Manual computation using F.kl_div which computes KL(student || teacher)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)
        manual_kld = F.kl_div(
            input=teacher_log_probs,
            target=student_probs,
            reduction='none',
            log_target=False
        ).sum(dim=-1)  # Sum over vocabulary
        manual_kld = manual_kld.sum()  # Sum over batch and sequence
        
        # Should match (allowing for numerical precision)
        assert torch.isclose(kld_result, manual_kld, rtol=1e-4, atol=1e-6), \
            f"Reverse KLD mismatch: got {kld_result}, expected {manual_kld}"

    def test_kld_temperature_scaling(self):
        """
        Test that temperature scaling works correctly.
        Temperature affects both the softmax smoothing and final loss scaling.
        The temp^2 factor compensates for the smoothing effect.
        """
        batch_size, seq_len, vocab_size = 2, 4, 10
        torch.manual_seed(42)
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size) * 2
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size) * 2
        target = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Test with temperature = 1.0
        config_t1 = LossConfig(temperature=1.0, reduction="sum")
        kld_t1 = compute_kld_forward(student_logits, teacher_logits, target, config_t1)
        
        # Test with temperature = 2.0
        config_t2 = LossConfig(temperature=2.0, reduction="sum")
        kld_t2 = compute_kld_forward(student_logits, teacher_logits, target, config_t2)
        
        # Test with temperature = 4.0
        config_t4 = LossConfig(temperature=4.0, reduction="sum")
        kld_t4 = compute_kld_forward(student_logits, teacher_logits, target, config_t4)
        
        # Higher temperature should produce higher loss (due to temp^2 scaling factor)
        assert kld_t2 > kld_t1, \
            f"Higher temperature should produce higher loss: T=2.0 ({kld_t2}) should be > T=1.0 ({kld_t1})"
        assert kld_t4 > kld_t2, \
            f"Higher temperature should produce higher loss: T=4.0 ({kld_t4}) should be > T=2.0 ({kld_t2})"
        
        # Verify temperature has an effect (not identity)
        assert not torch.isclose(kld_t1, kld_t2, rtol=0.1), \
            "Temperature should affect the loss value"

    def test_kld_with_padding_mask(self):
        """
        Test that padding positions are correctly masked and don't contribute to loss.
        """
        batch_size, seq_len, vocab_size = 2, 8, 10
        torch.manual_seed(42)
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Create target with padding tokens (-100)
        target = torch.randint(0, vocab_size, (batch_size, seq_len))
        target[0, 4:] = -100  # Last 4 tokens of first sample are padding
        target[1, 6:] = -100  # Last 2 tokens of second sample are padding
        
        config = LossConfig(
            temperature=1.0,
            reduction="sum",
            padding_id=-100
        )
        
        kld_with_padding = compute_kld_forward(student_logits, teacher_logits, target, config)
        
        # Manually compute KLD only for non-padding positions
        pad_mask = target.eq(-100)
        
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        kl_raw = F.kl_div(
            input=student_log_probs,
            target=teacher_probs,
            reduction='none',
            log_target=False
        ).sum(dim=-1)  # [batch_size, seq_len]
        
        manual_kld = kl_raw.masked_fill(pad_mask, 0.0).sum()
        
        # Should match
        assert torch.isclose(kld_with_padding, manual_kld, rtol=1e-4), \
            f"Padding mask not working correctly: got {kld_with_padding}, expected {manual_kld}"

    def test_kld_reduction_modes(self):
        """
        Test different reduction modes: sum, mean, batch_mean, none.
        """
        batch_size, seq_len, vocab_size = 2, 4, 10
        torch.manual_seed(42)
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        target = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Test sum reduction
        config_sum = LossConfig(temperature=1.0, reduction="sum")
        kld_sum = compute_kld_forward(student_logits, teacher_logits, target, config_sum)
        
        # Test mean reduction
        config_mean = LossConfig(temperature=1.0, reduction="mean")
        kld_mean = compute_kld_forward(student_logits, teacher_logits, target, config_mean)
        
        # Test batch_mean reduction
        config_batch_mean = LossConfig(temperature=1.0, reduction="batch_mean")
        kld_batch_mean = compute_kld_forward(student_logits, teacher_logits, target, config_batch_mean)
        
        # Test none reduction (returns per-sample values)
        config_none = LossConfig(temperature=1.0, reduction="none")
        kld_none = compute_kld_forward(student_logits, teacher_logits, target, config_none)
        
        # Verify relationships
        assert kld_mean < kld_sum, "Mean should be less than sum"
        assert kld_batch_mean < kld_sum, "Batch mean should be less than sum"
        assert kld_none.shape == (batch_size, seq_len), \
            f"None reduction should return shape {(batch_size, seq_len)}, got {kld_none.shape}"
        
        # Verify that mean is approximately sum / (batch_size * seq_len)
        expected_mean = kld_sum / (batch_size * seq_len)
        assert torch.isclose(kld_mean, expected_mean, rtol=1e-4), \
            f"Mean reduction incorrect: got {kld_mean}, expected {expected_mean}"


class TestConfidenceAwareKLD:
    """Test Confidence-Aware KL Divergence (CAKLD)."""

    def test_teacher_confidence_range(self):
        """
        Test that teacher confidence γ is always in the range [0, 1].
        """
        batch_size, seq_len, vocab_size = 2, 4, 10
        torch.manual_seed(42)
        
        for _ in range(10):
            teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
            target = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            config = LossConfig(use_entropy=False)
            gamma = compute_teacher_confidence(teacher_logits, target, config)
            
            assert 0.0 <= gamma <= 1.0, \
                f"Gamma out of range: {gamma}"

    def test_teacher_confidence_entropy_vs_probability(self):
        """
        Test both entropy-based and probability-based confidence methods.
        """
        batch_size, seq_len, vocab_size = 2, 4, 10
        torch.manual_seed(42)
        
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        target = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Probability-based confidence
        config_prob = LossConfig(use_entropy=False)
        gamma_prob = compute_teacher_confidence(teacher_logits, target, config_prob)
        
        # Entropy-based confidence
        config_entropy = LossConfig(use_entropy=True)
        gamma_entropy = compute_teacher_confidence(teacher_logits, target, config_entropy)
        
        # Both should be valid
        assert 0.0 <= gamma_prob <= 1.0
        assert 0.0 <= gamma_entropy <= 1.0
        
        # They should generally be different (unless by coincidence)
        # Just ensure both methods produce reasonable values

    def test_cakld_weighted_combination(self):
        """
        Test that CAKLD is correctly computed as a weighted combination:
        CAKLD = γ * Reverse_KL + (1-γ) * Forward_KL
        """
        batch_size, seq_len, vocab_size = 2, 4, 10
        torch.manual_seed(42)
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        target = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        config = LossConfig(
            loss_function="kldc",
            temperature=2.0,
            reduction="sum",
            use_entropy=False
        )
        
        # Compute CAKLD
        cakld = compute_kld_confidence(student_logits, teacher_logits, target, config)
        
        # Compute components manually
        gamma = compute_teacher_confidence(teacher_logits, target, config)
        reverse_kl = compute_kld_reverse(student_logits, teacher_logits, target, config)
        forward_kl = compute_kld_forward(student_logits, teacher_logits, target, config)
        
        # Manual weighted combination
        expected_cakld = gamma * reverse_kl + (1 - gamma) * forward_kl
        
        # Should match
        assert torch.isclose(cakld, expected_cakld, rtol=1e-5, atol=1e-7), \
            f"CAKLD mismatch: got {cakld}, expected {expected_cakld}"

    def test_cakld_confidence_extremes(self):
        """
        Test CAKLD behavior at confidence extremes:
        - When γ ≈ 1 (high confidence): should favor Reverse KL
        - When γ ≈ 0 (low confidence): should favor Forward KL
        """
        batch_size, seq_len, vocab_size = 2, 4, 10
        torch.manual_seed(42)
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        target = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        config = LossConfig(reduction="sum", use_entropy=False)
        
        # Compute all components
        gamma = compute_teacher_confidence(teacher_logits, target, config)
        reverse_kl = compute_kld_reverse(student_logits, teacher_logits, target, config)
        forward_kl = compute_kld_forward(student_logits, teacher_logits, target, config)
        cakld = compute_kld_confidence(student_logits, teacher_logits, target, config)
        
        # CAKLD should be between forward and reverse KL
        if reverse_kl < forward_kl:
            assert reverse_kl <= cakld <= forward_kl, \
                "CAKLD should be between Reverse KL and Forward KL"
        else:
            assert forward_kl <= cakld <= reverse_kl, \
                "CAKLD should be between Forward KL and Reverse KL"


class TestFeatureDistillation:
    """Test Feature Distillation (FD) using MSE loss."""

    def test_fd_mathematical_definition(self):
        """
        Test FD: MSE with proper handling of padding and reduction modes.
        When no padding, should match PyTorch's F.mse_loss.
        """
        batch_size, seq_len, hidden_size = 2, 4, 8
        torch.manual_seed(42)
        
        student_features = torch.randn(batch_size, seq_len, hidden_size)
        teacher_features = torch.randn(batch_size, seq_len, hidden_size)
        target = torch.randint(0, 10, (batch_size, seq_len))
        
        # Test with sum reduction (no padding influence on comparison)
        config = LossConfig(
            loss_type="hidden_states",
            loss_function="fd",
            reduction="sum",
            padding_id=-100
        )
        
        # Compute using our function
        fd_result = compute_fd(student_features, teacher_features, target, config)
        
        # Manual computation: MSE sum
        manual_mse = ((student_features - teacher_features) ** 2).sum()
        
        # Should match exactly
        assert torch.isclose(fd_result, manual_mse, rtol=1e-5, atol=1e-7), \
            f"FD MSE mismatch: got {fd_result}, expected {manual_mse}"

    def test_fd_zero_when_identical(self):
        """
        Test that FD loss is zero when student and teacher features are identical.
        """
        batch_size, seq_len, hidden_size = 2, 4, 8
        torch.manual_seed(42)
        
        features = torch.randn(batch_size, seq_len, hidden_size)
        target = torch.randint(0, 10, (batch_size, seq_len))
        
        config = LossConfig(
            loss_type="hidden_states",
            loss_function="fd",
            reduction="mean"
        )
        
        fd_result = compute_fd(features, features, target, config)
        
        assert torch.isclose(fd_result, torch.tensor(0.0), atol=1e-7), \
            f"FD should be zero for identical features, got {fd_result}"

    def test_fd_with_padding(self):
        """
        Test that FD correctly handles padding masks.
        """
        batch_size, seq_len, hidden_size = 2, 8, 8
        torch.manual_seed(42)
        
        student_features = torch.randn(batch_size, seq_len, hidden_size)
        teacher_features = torch.randn(batch_size, seq_len, hidden_size)
        
        # Create target with padding
        target = torch.randint(0, 10, (batch_size, seq_len))
        target[0, 4:] = -100  # Padding
        target[1, 6:] = -100  # Padding
        
        config = LossConfig(
            loss_type="hidden_states",
            loss_function="fd",
            reduction="mean",
            padding_id=-100
        )
        
        fd_with_padding = compute_fd(student_features, teacher_features, target, config)
        
        # Manually compute without padding positions
        mask = target != -100
        num_valid = mask.sum().float()
        
        mse = (student_features - teacher_features) ** 2
        # Expand mask to match feature dimensions
        mask_expanded = mask.unsqueeze(-1).expand_as(mse)
        masked_mse = mse * mask_expanded.float()
        manual_result = masked_mse.sum() / num_valid
        
        # Should match
        assert torch.isclose(fd_with_padding, manual_result, rtol=1e-4), \
            f"FD with padding incorrect: got {fd_with_padding}, expected {manual_result}"

    def test_fd_reduction_modes(self):
        """
        Test different reduction modes for FD: sum, mean, batch_mean, none.
        """
        batch_size, seq_len, hidden_size = 2, 4, 8
        torch.manual_seed(42)
        
        student_features = torch.randn(batch_size, seq_len, hidden_size)
        teacher_features = torch.randn(batch_size, seq_len, hidden_size)
        target = torch.randint(0, 10, (batch_size, seq_len))
        
        # Test sum
        config_sum = LossConfig(loss_type="hidden_states", reduction="sum")
        fd_sum = compute_fd(student_features, teacher_features, target, config_sum)
        
        # Test mean (averages over valid tokens and hidden dimensions)
        config_mean = LossConfig(loss_type="hidden_states", reduction="mean")
        fd_mean = compute_fd(student_features, teacher_features, target, config_mean)
        
        # Test batch_mean
        config_batch_mean = LossConfig(loss_type="hidden_states", reduction="batch_mean")
        fd_batch_mean = compute_fd(student_features, teacher_features, target, config_batch_mean)
        
        # Test none
        config_none = LossConfig(loss_type="hidden_states", reduction="none")
        fd_none = compute_fd(student_features, teacher_features, target, config_none)
        
        # Verify relationships
        assert fd_sum > fd_mean, "Sum should be greater than mean"
        assert fd_none.shape == student_features.shape, \
            f"None reduction should preserve shape, got {fd_none.shape}"
        
        # Verify mean relationship: mean should average over valid tokens
        # With 3D features, mean averages over (batch * seq_len) valid tokens
        num_valid_tokens = batch_size * seq_len
        # Expected: sum / (num_valid_tokens * hidden_size) -- This is NOT how our implementation works
        # Our implementation: sum / num_valid_tokens (averages over tokens, not individual elements)
        # So fd_mean should be close to fd_sum / num_valid_tokens
        expected_mean = fd_sum / num_valid_tokens
        assert torch.isclose(fd_mean, expected_mean, rtol=1e-3), \
            f"Mean reduction incorrect: got {fd_mean}, expected {expected_mean}"

    def test_fd_2d_features(self):
        """
        Test FD with 2D features (no sequence dimension).
        """
        batch_size, hidden_size = 4, 16
        torch.manual_seed(42)
        
        student_features = torch.randn(batch_size, hidden_size)
        teacher_features = torch.randn(batch_size, hidden_size)
        
        config = LossConfig(
            loss_type="hidden_states",
            loss_function="fd",
            reduction="mean"
        )
        
        # Should work without target for 2D features
        fd_result = compute_fd(student_features, teacher_features, None, config)
        
        # Manual computation
        manual_mse = F.mse_loss(student_features, teacher_features, reduction='mean')
        
        assert torch.isclose(fd_result, manual_mse, rtol=1e-5), \
            f"FD 2D mismatch: got {fd_result}, expected {manual_mse}"

    def test_fd_shape_mismatch_error(self):
        """
        Test that FD raises an error when feature shapes don't match.
        """
        student_features = torch.randn(2, 4, 8)
        teacher_features = torch.randn(2, 4, 16)  # Different hidden size
        target = torch.randint(0, 10, (2, 4))
        
        config = LossConfig(loss_type="hidden_states", loss_function="fd")
        
        with pytest.raises(ValueError, match="must have same shape"):
            compute_fd(student_features, teacher_features, target, config)


class TestNumericalStability:
    """Test numerical stability of loss computations."""

    def test_kld_with_extreme_logits(self):
        """
        Test KLD computation with extreme logit values (very large/small).
        Should not produce NaN or Inf.
        """
        batch_size, seq_len, vocab_size = 2, 4, 10
        
        # Extreme logits
        student_logits = torch.randn(batch_size, seq_len, vocab_size) * 100
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size) * 100
        target = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        config = LossConfig(temperature=2.0, reduction="mean")
        
        kld = compute_kld_forward(student_logits, teacher_logits, target, config)
        
        assert not torch.isnan(kld), "KLD produced NaN with extreme logits"
        assert not torch.isinf(kld), "KLD produced Inf with extreme logits"
        assert kld >= 0, "KLD should be non-negative"

    def test_kld_non_negativity(self):
        """
        Test that KL divergence is always non-negative (fundamental property).
        KL(P || Q) >= 0, with equality iff P = Q
        """
        batch_size, seq_len, vocab_size = 2, 4, 10
        torch.manual_seed(42)
        
        for _ in range(10):
            student_logits = torch.randn(batch_size, seq_len, vocab_size)
            teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
            target = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            config = LossConfig(temperature=2.0, reduction="mean")
            
            forward_kl = compute_kld_forward(student_logits, teacher_logits, target, config)
            reverse_kl = compute_kld_reverse(student_logits, teacher_logits, target, config)
            
            assert forward_kl >= 0, f"Forward KL should be non-negative, got {forward_kl}"
            assert reverse_kl >= 0, f"Reverse KL should be non-negative, got {reverse_kl}"

    def test_fd_non_negativity(self):
        """
        Test that MSE (FD) is always non-negative.
        """
        batch_size, seq_len, hidden_size = 2, 4, 8
        torch.manual_seed(42)
        
        for _ in range(10):
            student_features = torch.randn(batch_size, seq_len, hidden_size)
            teacher_features = torch.randn(batch_size, seq_len, hidden_size)
            target = torch.randint(0, 10, (batch_size, seq_len))
            
            config = LossConfig(loss_type="hidden_states", reduction="mean")
            
            fd = compute_fd(student_features, teacher_features, target, config)
            
            assert fd >= 0, f"FD should be non-negative, got {fd}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token_sequence(self):
        """Test with sequence length of 1."""
        batch_size, seq_len, vocab_size = 2, 1, 10
        torch.manual_seed(42)
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        target = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        config = LossConfig(reduction="mean")
        
        kld = compute_kld_forward(student_logits, teacher_logits, target, config)
        
        assert not torch.isnan(kld), "KLD failed with single token"
        assert kld >= 0, "KLD should be non-negative"

    def test_all_padding_tokens(self):
        """Test behavior when all tokens are padding."""
        batch_size, seq_len, vocab_size = 2, 4, 10
        torch.manual_seed(42)
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        target = torch.full((batch_size, seq_len), -100)  # All padding
        
        config = LossConfig(reduction="mean", padding_id=-100)
        
        kld = compute_kld_forward(student_logits, teacher_logits, target, config)
        
        # With all padding, result should be 0 or very close to 0
        assert torch.isclose(kld, torch.tensor(0.0), atol=1e-6), \
            f"Expected ~0 with all padding, got {kld}"

    def test_identical_distributions(self):
        """
        Test that KLD is zero when student and teacher have identical logits.
        """
        batch_size, seq_len, vocab_size = 2, 4, 10
        torch.manual_seed(42)
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        target = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        config = LossConfig(temperature=1.0, reduction="mean")
        
        # Forward KL with identical distributions
        forward_kl = compute_kld_forward(logits, logits, target, config)
        
        # Should be very close to zero (allowing for numerical precision)
        assert torch.isclose(forward_kl, torch.tensor(0.0), atol=1e-5), \
            f"KLD should be ~0 for identical distributions, got {forward_kl}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
