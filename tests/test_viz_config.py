"""
Tests for visualization configuration with flexible moment handling.

Tests:
- get_moment_label(): Label generation with fallbacks
- get_moment_color(): Color assignment with wrap-around
- get_moment_order(): Index resolution in sorted lists

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import pytest
from src.visualization.config import (
    get_moment_label,
    get_moment_color,
    get_moment_order,
    MOMENT_LABELS,
    MOMENT_COLORS,
)


class TestGetMomentLabel:
    """Tests for get_moment_label() function."""
    
    def test_known_labels(self):
        """Test that known moment labels are returned correctly."""
        assert get_moment_label('restingstate') == 'Resting State'
        assert get_moment_label('therapy') == 'Therapy Session'
    
    def test_auto_generation_single_word(self):
        """Test auto-generation for single word moments."""
        assert get_moment_label('baseline') == 'Baseline'
        assert get_moment_label('recovery') == 'Recovery'
        assert get_moment_label('intervention') == 'Intervention'
    
    def test_auto_generation_underscore(self):
        """Test auto-generation with underscores."""
        assert get_moment_label('post_intervention') == 'Post Intervention'
        assert get_moment_label('pre_stress') == 'Pre Stress'
        assert get_moment_label('stress_test') == 'Stress Test'
        assert get_moment_label('recovery_phase') == 'Recovery Phase'
    
    def test_config_override(self):
        """Test config override takes precedence."""
        config = {
            'visualization': {
                'moment_labels': {
                    'baseline': 'Custom Baseline Label',
                    'restingstate': 'Custom Resting'
                }
            }
        }
        assert get_moment_label('baseline', config) == 'Custom Baseline Label'
        assert get_moment_label('restingstate', config) == 'Custom Resting'
        # Unknown moment should still auto-generate
        assert get_moment_label('unknown_moment', config) == 'Unknown Moment'
    
    def test_fallback_chain(self):
        """Test the 3-tier fallback chain works correctly."""
        # 1. Config override (tested above)
        # 2. Known labels
        assert get_moment_label('therapy') == MOMENT_LABELS['therapy']
        # 3. Auto-generation
        assert get_moment_label('new_moment') == 'New Moment'


class TestGetMomentColor:
    """Tests for get_moment_color() function."""
    
    def test_string_moment_names(self):
        """Test color assignment for string moment names."""
        # Should return valid hex colors
        color1 = get_moment_color('baseline')
        color2 = get_moment_color('intervention')
        
        assert color1.startswith('#')
        assert len(color1) == 7
        assert color2.startswith('#')
        assert len(color2) == 7
    
    def test_color_consistency(self):
        """Test that same moment always gets same color."""
        moment = 'restingstate'
        color1 = get_moment_color(moment)
        color2 = get_moment_color(moment)
        assert color1 == color2
    
    def test_wrap_around_behavior(self):
        """Test that colors are assigned from palette (hash-based)."""
        # We have 8 colors in palette
        num_colors = len(MOMENT_COLORS)
        
        # Test moments beyond palette size
        moments = [f'moment_{i}' for i in range(num_colors + 3)]
        colors = [get_moment_color(m) for m in moments]
        
        # All colors should be from palette
        for i in range(len(moments)):
            assert colors[i] in MOMENT_COLORS
        
        # Same moment should always get same color (consistency)
        assert get_moment_color('moment_0') == get_moment_color('moment_0')
        assert get_moment_color('moment_5') == get_moment_color('moment_5')
    
    def test_integer_indices(self):
        """Test that integer indices still work (backward compatibility)."""
        color0 = get_moment_color(0)
        color1 = get_moment_color(1)
        
        assert color0 == MOMENT_COLORS[0]
        assert color1 == MOMENT_COLORS[1]


class TestGetMomentOrder:
    """Tests for get_moment_order() function."""
    
    def test_basic_ordering(self):
        """Test basic index lookup."""
        moments = ['baseline', 'intervention', 'recovery']
        
        assert get_moment_order('baseline', moments) == 0
        assert get_moment_order('intervention', moments) == 1
        assert get_moment_order('recovery', moments) == 2
    
    def test_alphabetical_sorting(self):
        """Test with alphabetically sorted list."""
        moments = sorted(['therapy', 'restingstate', 'baseline'])
        # After sorting: ['baseline', 'restingstate', 'therapy']
        
        assert get_moment_order('baseline', moments) == 0
        assert get_moment_order('restingstate', moments) == 1
        assert get_moment_order('therapy', moments) == 2
    
    def test_not_found(self):
        """Test that -1 is returned for missing moments."""
        moments = ['baseline', 'intervention']
        
        assert get_moment_order('nonexistent', moments) == -1
        assert get_moment_order('unknown', moments) == -1
    
    def test_empty_list(self):
        """Test with empty moments list."""
        assert get_moment_order('any_moment', []) == -1
    
    def test_single_moment(self):
        """Test with single moment."""
        moments = ['only_moment']
        assert get_moment_order('only_moment', moments) == 0
        assert get_moment_order('other', moments) == -1


class TestBackwardCompatibility:
    """Tests to ensure existing code still works."""
    
    def test_existing_moment_names(self):
        """Test that restingstate and therapy still work as expected."""
        existing_moments = ['restingstate', 'therapy']
        
        # Labels should be from MOMENT_LABELS dict
        assert get_moment_label('restingstate') == 'Resting State'
        assert get_moment_label('therapy') == 'Therapy Session'
        
        # Colors should work
        color_rest = get_moment_color('restingstate')
        color_therapy = get_moment_color('therapy')
        assert color_rest.startswith('#')
        assert color_therapy.startswith('#')
        
        # Order should work
        assert get_moment_order('restingstate', existing_moments) == 0
        assert get_moment_order('therapy', existing_moments) == 1
    
    def test_no_breaking_changes(self):
        """Verify no breaking changes to existing behavior."""
        # get_moment_color should accept both int and str
        assert isinstance(get_moment_color(0), str)
        assert isinstance(get_moment_color('moment'), str)
        
        # get_moment_label should work without config
        assert isinstance(get_moment_label('any_moment'), str)
        
        # get_moment_order should handle edge cases gracefully
        assert get_moment_order('x', []) == -1


class TestIntegrationScenarios:
    """Test realistic usage scenarios."""
    
    def test_two_moments_scenario(self):
        """Test current production scenario (restingstate + therapy)."""
        moments = ['restingstate', 'therapy']
        
        labels = [get_moment_label(m) for m in moments]
        colors = [get_moment_color(m) for m in moments]
        orders = [get_moment_order(m, moments) for m in moments]
        
        assert labels == ['Resting State', 'Therapy Session']
        assert len(colors) == 2
        assert all(c.startswith('#') for c in colors)
        assert orders == [0, 1]
    
    def test_three_moments_scenario(self):
        """Test extended scenario with 3 moments."""
        moments = sorted(['baseline', 'intervention', 'recovery'])
        
        labels = [get_moment_label(m) for m in moments]
        colors = [get_moment_color(m) for m in moments]
        orders = [get_moment_order(m, moments) for m in moments]
        
        assert labels == ['Baseline', 'Intervention', 'Recovery']
        assert len(colors) == 3
        # Note: Hash collisions possible, so we can't guarantee all unique
        # But each moment should get a valid color from palette
        assert all(c in MOMENT_COLORS for c in colors)
        assert orders == [0, 1, 2]
    
    def test_custom_names_scenario(self):
        """Test with custom moment names."""
        moments = sorted(['pre_stress', 'stress_test', 'post_stress'])
        
        labels = [get_moment_label(m) for m in moments]
        
        assert labels == ['Post Stress', 'Pre Stress', 'Stress Test']
        assert all(isinstance(l, str) for l in labels)
        assert all(' ' in l for l in labels)  # All have spaces
    
    def test_many_moments_scenario(self):
        """Test with more moments than colors available."""
        moments = [f'phase_{i}' for i in range(10)]
        
        colors = [get_moment_color(m) for m in moments]
        
        # Should have 10 colors
        assert len(colors) == 10
        
        # All colors should be from palette (hash-based assignment)
        assert all(c in MOMENT_COLORS for c in colors)
        
        # Consistency check: same moment always gets same color
        for moment in moments:
            color1 = get_moment_color(moment)
            color2 = get_moment_color(moment)
            assert color1 == color2
