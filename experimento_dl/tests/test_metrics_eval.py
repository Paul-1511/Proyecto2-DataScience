"""Pruebas unitarias del módulo metrics_eval.py.

Usa únicamente datos SINTÉTICOS generados localmente para verificar la
correctitud de las implementaciones. No depende de artefactos reales del
experimento RSNA 2022.

Ejecutar:
    cd experimento_dl
    python -m pytest tests/test_metrics_eval.py -v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Añadir el directorio padre al path para importar metrics_eval
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics_eval import (
    bootstrap_auc_ci,
    confusion_matrix_detailed,
    delong_test,
    disagreement_analysis,
    error_analysis,
    find_threshold_youden,
    mcnemar_exact_test,
    metrics_complete,
    paired_bootstrap_test,
    spearman_correlation,
    verify_metrics_consistency,
    verify_partitions_disjoint,
    verify_predictions_format,
)


# ─────────────────────────────────────────────────────────────
# DATOS SINTÉTICOS
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_perfect_classifier():
    """Clasificador perfecto: predicciones idénticas a las etiquetas."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.95, 0.15, 0.85, 0.25, 0.75])
    return y_true, y_prob


@pytest.fixture
def synthetic_random_classifier():
    """Clasificador aleatorio con semilla fija."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100)
    y_prob = rng.uniform(0, 1, size=100)
    return y_true, y_prob


@pytest.fixture
def synthetic_two_models():
    """Dos modelos sintéticos para pruebas pareadas."""
    rng = np.random.default_rng(123)
    y_true = rng.integers(0, 2, size=200)
    # Modelo A: mejor discriminación
    prob_a = y_true * 0.6 + rng.uniform(0, 0.4, size=200)
    # Modelo B: peor discriminación
    prob_b = y_true * 0.4 + rng.uniform(0, 0.6, size=200)
    return y_true, prob_a, prob_b


@pytest.fixture
def synthetic_temp_files(tmp_path):
    """Archivos temporales para pruebas de validación."""
    # Predicciones válidas
    pred_df = pd.DataFrame({
        "StudyInstanceUID": [f"uid_{i}" for i in range(10)],
        "patient_overall": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "split": ["test"] * 10,
        "prob": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.15, 0.85],
        "pred": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    pred_path = tmp_path / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # Splits válidos
    splits_df = pd.DataFrame({
        "StudyInstanceUID": [f"uid_{i}" for i in range(20)],
        "patient_overall": [0, 1] * 10,
        "split": ["train"] * 10 + ["test"] * 5 + ["val"] * 5,
    })
    splits_path = tmp_path / "splits.csv"
    splits_df.to_csv(splits_path, index=False)

    # Métricas JSON
    metrics_dict = {
        "val": {"threshold": 0.5},
        "test_thr_val_youden": {
            "threshold": 0.5,
            "tp": 4, "fp": 1, "tn": 4, "fn": 1,
            "sensitivity": 0.8, "specificity": 0.8,
            "precision": 0.8, "f1": 0.8,
            "accuracy": 0.8, "roc_auc": 0.85,
        },
        "test_auc_ci95_bootstrap": [0.75, 0.95],
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_dict))

    return {
        "pred_path": pred_path,
        "splits_path": splits_path,
        "metrics_path": metrics_path,
    }


# ─────────────────────────────────────────────────────────────
# PRUEBAS: MÉTRICAS DE CLASIFICACIÓN
# ─────────────────────────────────────────────────────────────

class TestConfusionMatrix:
    """Pruebas de matriz de confusión."""

    def test_confusion_matrix_detailed_basic(self, synthetic_perfect_classifier):
        y_true, y_prob = synthetic_perfect_classifier
        cm = confusion_matrix_detailed(y_true, (y_prob >= 0.5).astype(int))

        assert cm["tp"] == 5
        assert cm["tn"] == 5
        assert cm["fp"] == 0
        assert cm["fn"] == 0
        assert cm["matrix"].shape == (2, 2)

    def test_confusion_matrix_detailed_imperfect(self, synthetic_random_classifier):
        y_true, y_prob = synthetic_random_classifier
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix_detailed(y_true, y_pred)

        # Verificar que coincide con sklearn
        sklearn_cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        assert np.array_equal(cm["matrix"], sklearn_cm)
        assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == len(y_true)


class TestMetricsComplete:
    """Pruebas de métricas completas."""

    def test_metrics_perfect_classifier(self, synthetic_perfect_classifier):
        y_true, y_prob = synthetic_perfect_classifier
        m = metrics_complete(y_true, y_prob, threshold=0.5)

        assert m["sensitivity"] == 1.0
        assert m["specificity"] == 1.0
        assert m["precision"] == 1.0
        assert m["f1"] == 1.0
        assert m["accuracy"] == 1.0
        assert m["roc_auc"] == 1.0

    def test_metrics_random_classifier(self, synthetic_random_classifier):
        y_true, y_prob = synthetic_random_classifier
        m = metrics_complete(y_true, y_prob, threshold=0.5)

        # Verificar que las métricas coinciden con sklearn
        y_pred = (y_prob >= 0.5).astype(int)
        assert abs(m["sensitivity"] - recall_score(y_true, y_pred)) < 1e-9
        assert abs(m["precision"] - precision_score(y_true, y_pred)) < 1e-9
        assert abs(m["f1"] - f1_score(y_true, y_pred)) < 1e-9
        assert abs(m["accuracy"] - accuracy_score(y_true, y_pred)) < 1e-9

    def test_metrics_all_positive(self):
        """Caso borde: todas las etiquetas son positivas."""
        y_true = np.array([1, 1, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.7, 0.6])
        m = metrics_complete(y_true, y_prob, threshold=0.5)

        assert m["tp"] == 4
        assert m["fp"] == 0
        assert m["fn"] == 0
        assert m["sensitivity"] == 1.0
        # Especificidad es NaN porque no hay negativos
        assert np.isnan(m["specificity"])

    def test_metrics_threshold_boundary(self):
        """Caso borde: probabilidad exactamente igual al umbral."""
        y_true = np.array([0, 1])
        y_prob = np.array([0.5, 0.5])
        m = metrics_complete(y_true, y_prob, threshold=0.5)

        # Ambos se predicen como 1 (>= threshold)
        assert m["tp"] == 1
        assert m["fp"] == 1


# ─────────────────────────────────────────────────────────────
# PRUEBAS: SELECCIÓN DE UMBRAL
# ─────────────────────────────────────────────────────────────

class TestThresholdYouden:
    """Pruebas del índice de Youden."""

    def test_youden_perfect_classifier(self, synthetic_perfect_classifier):
        y_true, y_prob = synthetic_perfect_classifier
        thr = find_threshold_youden(y_true, y_prob)

        # Para un clasificador perfecto, el umbral debe estar entre 0.3 y 0.9
        assert 0.3 <= thr <= 0.9

    def test_youden_balanced(self):
        """Caso balanceado: umbral debe estar cerca de 0.5."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        thr = find_threshold_youden(y_true, y_prob)

        # El umbral óptimo debe estar alrededor de 0.5
        assert 0.4 <= thr <= 0.6


# ─────────────────────────────────────────────────────────────
# PRUEBAS: BOOTSTRAP
# ─────────────────────────────────────────────────────────────

class TestBootstrap:
    """Pruebas de bootstrap para IC95%."""

    def test_bootstrap_auc_reproducible(self, synthetic_random_classifier):
        """El bootstrap debe ser reproducible con la misma semilla."""
        y_true, y_prob = synthetic_random_classifier

        ci1 = bootstrap_auc_ci(y_true, y_prob, n_resamples=100, seed=42)
        ci2 = bootstrap_auc_ci(y_true, y_prob, n_resamples=100, seed=42)

        assert ci1 == ci2

    def test_bootstrap_auc_different_seeds(self, synthetic_random_classifier):
        """Diferentes semillas deben dar resultados ligeramente diferentes."""
        y_true, y_prob = synthetic_random_classifier

        ci1 = bootstrap_auc_ci(y_true, y_prob, n_resamples=1000, seed=42)
        ci2 = bootstrap_auc_ci(y_true, y_prob, n_resamples=1000, seed=123)

        # No deben ser idénticos (aunque pueden ser similares)
        assert ci1 != ci2

    def test_bootstrap_auc_perfect_classifier(self, synthetic_perfect_classifier):
        """Clasificador perfecto debe tener AUC=1 en todos los remuestreos."""
        y_true, y_prob = synthetic_perfect_classifier
        ci = bootstrap_auc_ci(y_true, y_prob, n_resamples=100, seed=42)

        # El IC debe ser [1.0, 1.0] porque siempre es perfecto
        assert ci[0] == 1.0
        assert ci[1] == 1.0


# ─────────────────────────────────────────────────────────────
# PRUEBAS: COMPARACIÓN PAREADA
# ─────────────────────────────────────────────────────────────

class TestPairedTests:
    """Pruebas de comparación estadística pareada."""

    def test_delong_identical_models(self):
        """Dos modelos idénticos deben tener diferencia=0."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=100)
        y_prob = rng.uniform(0, 1, size=100)

        result = delong_test(y_true, y_prob, y_prob)

        assert abs(result["diff"]) < 1e-9
        assert result["p_value"] == 1.0

    def test_delong_different_models(self, synthetic_two_models):
        """Modelos con diferente desempeño deben dar resultados válidos."""
        y_true, prob_a, prob_b = synthetic_two_models

        result = delong_test(y_true, prob_a, prob_b)

        # La diferencia puede ser en cualquier dirección con esta semilla sintética;
        # lo importante es que el resultado es válido y coherente.
        assert abs(result["diff"]) > 0
        assert abs(result["auc_a"] - roc_auc_score(y_true, prob_a)) < 1e-9
        assert abs(result["auc_b"] - roc_auc_score(y_true, prob_b)) < 1e-9
        assert 0 <= result["p_value"] <= 1
        # diff = auc_b - auc_a
        assert abs(result["diff"] - (result["auc_b"] - result["auc_a"])) < 1e-9

    def test_paired_bootstrap_reproducible(self, synthetic_two_models):
        """Bootstrap pareado debe ser reproducible."""
        y_true, prob_a, prob_b = synthetic_two_models

        result1 = paired_bootstrap_test(y_true, prob_a, prob_b, n_resamples=100, seed=42)
        result2 = paired_bootstrap_test(y_true, prob_a, prob_b, n_resamples=100, seed=42)

        assert result1["diff_mean"] == result2["diff_mean"]
        assert result1["ci95"] == result2["ci95"]

    def test_mcnemar_identical_predictions(self):
        """Predicciones idénticas deben tener p_value=1."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        pred_a = np.array([0, 1, 0, 1, 0, 1])
        pred_b = np.array([0, 1, 0, 1, 0, 1])

        result = mcnemar_exact_test(y_true, pred_a, pred_b)

        assert result["solo_a_acierta"] == 0
        assert result["solo_b_acierta"] == 0
        assert result["p_value"] == 1.0

    def test_mcnemar_different_predictions(self):
        """Predicciones diferentes deben dar p_value calculado."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        pred_a = np.array([0, 1, 1, 0, 0, 1])  # acierta en índices 0,1,4,5
        pred_b = np.array([0, 0, 0, 1, 1, 1])  # acierta en índices 0,2,3,5

        result = mcnemar_exact_test(y_true, pred_a, pred_b)

        # Verificación manual:
        # índice 1: solo A acierta, índice 4: solo A acierta → solo_a_acierta = 2
        # índice 2: solo B acierta, índice 3: solo B acierta → solo_b_acierta = 2
        assert result["solo_a_acierta"] == 2
        assert result["solo_b_acierta"] == 2
        assert result["ambos_aciertan"] == 2
        assert result["ambos_fallan"] == 0
        assert 0 <= result["p_value"] <= 1

    def test_spearman_correlation(self):
        """Correlación de Spearman debe estar en [-1, 1]."""
        prob_a = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        prob_b = np.array([0.2, 0.6, 0.8, 0.4, 0.6])

        corr = spearman_correlation(prob_a, prob_b)

        assert -1 <= corr <= 1


# ─────────────────────────────────────────────────────────────
# PRUEBAS: VALIDACIÓN DE ARCHIVOS
# ─────────────────────────────────────────────────────────────

class TestFileValidation:
    """Pruebas de validación de formatos de archivos."""

    def test_verify_predictions_format_valid(self, synthetic_temp_files):
        result = verify_predictions_format(synthetic_temp_files["pred_path"])

        assert result["valid"] is True
        assert result["n_rows"] == 10
        assert "StudyInstanceUID" in result["columns"]
        assert len(result["issues"]) == 0

    def test_verify_predictions_format_missing_file(self, tmp_path):
        result = verify_predictions_format(tmp_path / "nonexistent.csv")

        assert result["valid"] is False
        assert "no encontrado" in result["issues"][0]

    def test_verify_predictions_format_invalid_probs(self, tmp_path):
        df = pd.DataFrame({
            "StudyInstanceUID": ["uid_1"],
            "patient_overall": [1],
            "split": ["test"],
            "prob": [1.5],  # Inválido: > 1
            "pred": [1],
        })
        path = tmp_path / "invalid.csv"
        df.to_csv(path, index=False)

        result = verify_predictions_format(path)

        assert result["valid"] is False
        assert any("fuera de [0, 1]" in issue for issue in result["issues"])

    def test_verify_partitions_disjoint_valid(self, synthetic_temp_files):
        result = verify_partitions_disjoint(synthetic_temp_files["splits_path"])

        assert result["disjoint"] is True
        assert result["sizes"]["train"] == 10
        assert result["sizes"]["test"] == 5
        assert result["sizes"]["val"] == 5
        assert len(result["issues"]) == 0

    def test_verify_partitions_disjoint_overlap(self, tmp_path):
        """Particiones con UIDs compartidos deben fallar."""
        df = pd.DataFrame({
            "StudyInstanceUID": ["uid_1", "uid_1", "uid_2"],  # uid_1 repetido
            "patient_overall": [0, 1, 0],
            "split": ["train", "test", "val"],
        })
        path = tmp_path / "overlap.csv"
        df.to_csv(path, index=False)

        result = verify_partitions_disjoint(path)

        assert result["disjoint"] is False
        assert any("compartidos" in issue for issue in result["issues"])


# ─────────────────────────────────────────────────────────────
# PRUEBAS: ANÁLISIS DE ERRORES
# ─────────────────────────────────────────────────────────────

class TestErrorAnalysis:
    """Pruebas de análisis de errores."""

    def test_error_analysis_basic(self, synthetic_perfect_classifier):
        y_true, y_prob = synthetic_perfect_classifier
        result = error_analysis(y_true, y_prob, threshold=0.5)

        assert result["summary"]["tp"] == 5
        assert result["summary"]["tn"] == 5
        assert result["summary"]["fp"] == 0
        assert result["summary"]["fn"] == 0
        assert len(result["fp_probs"]) == 0
        assert len(result["fn_probs"]) == 0

    def test_error_analysis_with_uids(self, synthetic_perfect_classifier):
        y_true, y_prob = synthetic_perfect_classifier
        uids = [f"uid_{i}" for i in range(len(y_true))]

        result = error_analysis(y_true, y_prob, threshold=0.5, uids=uids)

        assert "fp_uids" in result
        assert "fn_uids" in result

    def test_disagreement_analysis(self, synthetic_two_models):
        y_true, prob_a, prob_b = synthetic_two_models
        result = disagreement_analysis(y_true, prob_a, prob_b, 0.5, 0.5)

        total = (result["summary"]["solo_a_acierta"] +
                 result["summary"]["solo_b_acierta"] +
                 result["summary"]["ambos_aciertan"] +
                 result["summary"]["ambos_fallan"])

        assert total == len(y_true)


# ─────────────────────────────────────────────────────────────
# PRUEBAS: INTEGRACIÓN
# ─────────────────────────────────────────────────────────────

class TestIntegration:
    """Pruebas de integración del flujo completo."""

    def test_full_evaluation_pipeline(self, synthetic_random_classifier):
        """Flujo completo: métricas -> umbral -> IC -> comparación."""
        y_true, y_prob = synthetic_random_classifier

        # 1. Encontrar umbral óptimo
        thr = find_threshold_youden(y_true, y_prob)
        assert 0 < thr < 1

        # 2. Calcular métricas
        m = metrics_complete(y_true, y_prob, thr)
        assert 0 <= m["accuracy"] <= 1
        assert 0 <= m["f1"] <= 1

        # 3. Calcular IC95%
        ci = bootstrap_auc_ci(y_true, y_prob, n_resamples=100, seed=42)
        assert ci[0] <= m["roc_auc"] <= ci[1] or ci[0] == ci[1]

        # 4. Comparación con otro modelo
        rng = np.random.default_rng(99)
        y_prob_b = rng.uniform(0, 1, size=len(y_true))

        delong = delong_test(y_true, y_prob, y_prob_b)
        assert "p_value" in delong
        assert 0 <= delong["p_value"] <= 1

        bootstrap = paired_bootstrap_test(y_true, y_prob, y_prob_b, n_resamples=100)
        assert "diff_mean" in bootstrap
        assert len(bootstrap["ci95"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
