"""
Breakdown prediction model for conversation analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import logging

logger = logging.getLogger(__name__)


class BreakdownPredictor:
    """
    Predicts conversation breakdown using trajectory features and phase patterns.
    """
    
    def __init__(self):
        """Initialize breakdown predictor."""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
    def extract_features(self,
                        ensemble_embeddings: Dict[str, np.ndarray],
                        trajectory_metrics: Dict,
                        phase_info: Dict,
                        lookback_turns: int = 20) -> Dict[str, float]:
        """
        Extract predictive features from conversation data.
        
        Args:
            ensemble_embeddings: Embeddings from each model
            trajectory_metrics: Trajectory metrics for each model
            phase_info: Phase detection results
            lookback_turns: Number of recent turns to analyze
            
        Returns:
            Dictionary of features
        """
        features = {}
        n_messages = len(next(iter(ensemble_embeddings.values())))
        
        # Recent trajectory features
        if n_messages >= lookback_turns:
            for model_name, embeddings in ensemble_embeddings.items():
                recent_features = self._extract_recent_features(
                    embeddings[-lookback_turns:],
                    model_name
                )
                features.update(recent_features)
                
        # Phase-related features
        phase_features = self._extract_phase_features(
            phase_info['detected_phases'],
            n_messages,
            lookback_turns
        )
        features.update(phase_features)
        
        # Ensemble agreement features
        features['ensemble_agreement_mean'] = phase_info['ensemble_agreement'].get('mean_correlation', 0)
        features['ensemble_agreement_min'] = phase_info['ensemble_agreement'].get('min_correlation', 0)
        features['ensemble_agreement_std'] = phase_info['ensemble_agreement'].get('std_correlation', 0)
        
        # Cross-model consistency
        consistency_features = self._extract_consistency_features(
            trajectory_metrics,
            n_messages,
            lookback_turns
        )
        features.update(consistency_features)
        
        return features
        
    def _extract_recent_features(self, 
                               recent_embeddings: np.ndarray,
                               model_name: str) -> Dict[str, float]:
        """Extract features from recent trajectory segment."""
        features = {}
        
        # Velocities
        velocities = []
        for i in range(1, len(recent_embeddings)):
            v = np.linalg.norm(recent_embeddings[i] - recent_embeddings[i-1])
            velocities.append(v)
            
        if velocities:
            features[f'{model_name}_recent_velocity_mean'] = np.mean(velocities)
            features[f'{model_name}_recent_velocity_std'] = np.std(velocities)
            features[f'{model_name}_recent_velocity_max'] = np.max(velocities)
            features[f'{model_name}_recent_velocity_trend'] = (
                (velocities[-1] - velocities[0]) / len(velocities) 
                if len(velocities) > 1 else 0
            )
            
        # Accelerations
        if len(velocities) > 1:
            accelerations = np.diff(velocities)
            features[f'{model_name}_recent_acceleration_mean'] = np.mean(np.abs(accelerations))
            features[f'{model_name}_recent_acceleration_max'] = np.max(np.abs(accelerations))
            
        # Curvature
        if len(recent_embeddings) >= 3:
            curvatures = []
            for i in range(1, len(recent_embeddings) - 1):
                curv = self._calculate_curvature(
                    recent_embeddings[i-1],
                    recent_embeddings[i],
                    recent_embeddings[i+1]
                )
                curvatures.append(curv)
                
            if curvatures:
                features[f'{model_name}_recent_curvature_mean'] = np.mean(curvatures)
                features[f'{model_name}_recent_curvature_max'] = np.max(curvatures)
                
        return features
        
    def _calculate_curvature(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate curvature at p2."""
        v1 = p2 - p1
        v2 = p3 - p2
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 < 1e-8 or norm_v2 < 1e-8:
            return 0
            
        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        avg_length = (norm_v1 + norm_v2) / 2
        return angle / avg_length if avg_length > 0 else 0
        
    def _extract_phase_features(self,
                              phases: List[Dict],
                              n_messages: int,
                              lookback_turns: int) -> Dict[str, float]:
        """Extract phase-related features."""
        features = {
            'total_phases': len(phases),
            'phases_per_turn': len(phases) / n_messages if n_messages > 0 else 0
        }
        
        # Recent phase activity
        recent_phases = [p for p in phases if p['turn'] >= n_messages - lookback_turns]
        features['recent_phase_count'] = len(recent_phases)
        features['recent_phase_density'] = len(recent_phases) / lookback_turns
        
        if recent_phases:
            features['recent_phase_magnitude_mean'] = np.mean([p['magnitude'] for p in recent_phases])
            features['recent_phase_confidence_mean'] = np.mean([p['confidence'] for p in recent_phases])
            features['turns_since_last_phase'] = n_messages - recent_phases[-1]['turn']
        else:
            features['recent_phase_magnitude_mean'] = 0
            features['recent_phase_confidence_mean'] = 0
            features['turns_since_last_phase'] = lookback_turns
            
        # Phase type distribution
        phase_types = [p.get('type', 'unknown') for p in phases]
        for phase_type in ['opening', 'closing', 'topic_shift', 'major_transition', 'development']:
            features[f'phase_type_{phase_type}_count'] = phase_types.count(phase_type)
            
        return features
        
    def _extract_consistency_features(self,
                                    trajectory_metrics: Dict,
                                    n_messages: int,
                                    lookback_turns: int) -> Dict[str, float]:
        """Extract cross-model consistency features."""
        features = {}
        
        # Get velocity consistency
        if 'consistency' in trajectory_metrics:
            consistency = trajectory_metrics['consistency']
            features['velocity_consistency'] = consistency.get('velocity_correlation', 0)
            features['velocity_consistency_std'] = consistency.get('velocity_correlation_std', 0)
            
        # Calculate divergence in recent velocities
        model_velocities = []
        for model_name, metrics in trajectory_metrics.items():
            if model_name != 'consistency' and 'velocities' in metrics:
                velocities = metrics['velocities']
                if len(velocities) >= lookback_turns:
                    model_velocities.append(velocities[-lookback_turns:])
                    
        if len(model_velocities) > 1:
            # Variance across models
            velocity_variance = np.var(model_velocities, axis=0)
            features['cross_model_velocity_variance'] = np.mean(velocity_variance)
            features['cross_model_velocity_divergence'] = velocity_variance[-1] if len(velocity_variance) > 0 else 0
            
        return features
        
    def train(self,
             labeled_data: List[Tuple[Dict, bool]],
             verbose: bool = True) -> Dict:
        """
        Train breakdown prediction model.
        
        Args:
            labeled_data: List of (features, is_breakdown) tuples
            verbose: Whether to print training progress
            
        Returns:
            Training metrics
        """
        if not labeled_data:
            raise ValueError("No training data provided")
            
        # Extract features and labels
        X = []
        y = []
        
        for features, label in labeled_data:
            if self.feature_names is None:
                self.feature_names = sorted(features.keys())
                
            # Convert to array using consistent order
            feature_vector = [features.get(name, 0) for name in self.feature_names]
            X.append(feature_vector)
            y.append(label)
            
        X = np.array(X)
        y = np.array(y)
        
        # Handle missing values
        X = np.nan_to_num(X, 0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate metrics
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        metrics = {
            'n_samples': len(X),
            'n_breakdowns': np.sum(y),
            'breakdown_rate': np.mean(y)
        }
        
        if len(np.unique(y)) > 1:
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
            
            precision, recall, _ = precision_recall_curve(y, y_pred_proba)
            metrics['pr_auc'] = auc(recall, precision)
        else:
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
            
        if verbose:
            logger.info(f"Training complete!")
            logger.info(f"  Samples: {metrics['n_samples']} ({metrics['n_breakdowns']} breakdowns)")
            logger.info(f"  Features: {len(self.feature_names)}")
            if metrics['roc_auc'] is not None:
                logger.info(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
                logger.info(f"  PR-AUC: {metrics['pr_auc']:.3f}")
                
            # Top features
            importance = np.abs(self.model.coef_[0])
            top_indices = np.argsort(importance)[-10:][::-1]
            
            logger.info("\n  Top 10 predictive features:")
            for idx in top_indices:
                logger.info(f"    - {self.feature_names[idx]}: {importance[idx]:.3f}")
                
        return metrics
        
    def predict(self,
               features: Dict[str, float],
               return_confidence: bool = True) -> Dict:
        """
        Predict breakdown probability.
        
        Args:
            features: Feature dictionary
            return_confidence: Whether to include confidence score
            
        Returns:
            Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
            
        # Convert to array
        X = np.array([features.get(name, 0) for name in self.feature_names]).reshape(1, -1)
        
        # Handle missing values
        X = np.nan_to_num(X, 0)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prob = self.model.predict_proba(X_scaled)[0, 1]
        prediction = self.model.predict(X_scaled)[0]
        
        result = {
            'probability': prob,
            'prediction': bool(prediction)
        }
        
        if return_confidence:
            # Confidence based on distance from decision boundary
            decision_score = self.model.decision_function(X_scaled)[0]
            confidence = 1 / (1 + np.exp(-abs(decision_score)))
            result['confidence'] = confidence
            
            # Contributing factors
            feature_importance = self.model.coef_[0]
            top_indices = np.argsort(np.abs(feature_importance * X_scaled[0]))[-5:][::-1]
            
            contributing_factors = []
            for idx in top_indices:
                if abs(feature_importance[idx]) > 0.1:
                    contributing_factors.append({
                        'feature': self.feature_names[idx],
                        'value': features.get(self.feature_names[idx], 0),
                        'importance': feature_importance[idx]
                    })
                    
            result['contributing_factors'] = contributing_factors
            
        return result
        
    def predict_with_lookahead(self,
                             features_sequence: List[Dict[str, float]],
                             max_lookahead: int = 30,
                             step: int = 5) -> List[Dict]:
        """
        Generate predictions for multiple lookahead windows.
        
        Args:
            features_sequence: Sequence of feature dictionaries
            max_lookahead: Maximum turns to look ahead
            step: Step size for lookahead windows
            
        Returns:
            List of predictions at different lookaheads
        """
        predictions = []
        
        for n_ahead in range(step, min(max_lookahead + 1, len(features_sequence) + 1), step):
            # Use features up to n_ahead
            current_features = features_sequence[min(n_ahead - 1, len(features_sequence) - 1)]
            
            pred = self.predict(current_features)
            pred['n_turns_ahead'] = n_ahead
            predictions.append(pred)
            
        return predictions
        
    def get_feature_importance(self) -> List[Tuple[str, float]]:
        """Get feature importance from trained model."""
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        importance = np.abs(self.model.coef_[0])
        
        return sorted(
            [(name, imp) for name, imp in zip(self.feature_names, importance)],
            key=lambda x: x[1],
            reverse=True
        )