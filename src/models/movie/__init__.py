"""
CineSync v2 - Movie-Specific Recommendation Models

This module contains 14 specialized recommendation models designed for movie-specific
features like franchises, director styles, cinematic universes, awards predictions, etc.

These models complement the content-agnostic models by leveraging movie-specific patterns
that don't apply to TV shows.
"""

# Franchise and Sequel Models
from .franchise_sequence import FranchiseSequenceModel

# Director and Creator Models
from .director_auteur import DirectorAuteurModel

# Cinematic Universe Models
from .cinematic_universe import CinematicUniverseModel

# Awards and Prestige Models
from .awards_prediction import AwardsPredictionModel

# Runtime and Pacing Models
from .runtime_preference import RuntimePreferenceModel

# Era and Style Models
from .era_style import EraStyleModel

# Critical Reception Models
from .critic_audience import CriticAudienceModel

# Remake and Adaptation Models
from .remake_connection import RemakeConnectionModel

# Cast and Collaboration Models
from .actor_collaboration import ActorCollaborationModel

# Studio and Production Models
from .studio_fingerprint import StudioFingerprintModel

# Source Material Models
from .adaptation_source import AdaptationSourceModel

# International Cinema Models
from .international_cinema import InternationalCinemaModel

# Narrative Structure Models
from .narrative_complexity import NarrativeComplexityModel

# Context-Aware Models
from .viewing_context import ViewingContextModel

__all__ = [
    'FranchiseSequenceModel',
    'DirectorAuteurModel',
    'CinematicUniverseModel',
    'AwardsPredictionModel',
    'RuntimePreferenceModel',
    'EraStyleModel',
    'CriticAudienceModel',
    'RemakeConnectionModel',
    'ActorCollaborationModel',
    'StudioFingerprintModel',
    'AdaptationSourceModel',
    'InternationalCinemaModel',
    'NarrativeComplexityModel',
    'ViewingContextModel',
]

__version__ = '1.0.0'
