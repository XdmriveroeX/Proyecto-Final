import sys
import json
import math
from datetime import datetime
import gradio as gr

# --- Constantes de Configuración ---

TASK_CATEGORIES = {
    "Programación": "coding_average",
    "Matemáticas": "math_average",
    "Análisis de Datos": "data_analysis_average",
    "Ciencia": "science",
    "Escritura": "writing",
    "Escritura Creativa": "creative_writing",
}
DIFFICULTY_LEVELS = {"Fácil": "facil", "Media": "media", "Difícil": "dificil"}
RELATIVE_VPI_THRESHOLD = 0.05

# --- Clases del Modelo de Decisión (Sin cambios en la lógica) ---

class ProbabilisticQualityModel:
    """
    Modela la incertidumbre sobre la calidad de la respuesta.
    Calcula la utilidad esperada de la calidad basado en la categoría y dificultad de la tarea.
    """
    QUALITY_STATES = {'Excelente': 1.0, 'Buena': 0.7, 'Regular': 0.2, 'Mala': 0.0}

    def get_expected_quality_utility(self, category, difficulty, model_data):
        """
        Calcula la Utilidad Esperada de la calidad para un modelo y tarea dados.
        """
        base_prob_good_or_better = model_data.get(category, 50) / 100.0

        if difficulty == 'facil':
            adjustment_factor = 1.15
        elif difficulty == 'dificil':
            adjustment_factor = 0.85
        else:
            adjustment_factor = 1.0

        prob_good_or_better = min(max(base_prob_good_or_better * adjustment_factor, 0), 0.98)

        prob_excelente = prob_good_or_better / 3.0
        prob_buena = prob_good_or_better * (2/3.0)
        remaining_prob = 1.0 - prob_excelente - prob_buena
        prob_regular = remaining_prob * 0.7
        prob_mala = remaining_prob * 0.3

        probabilities = {
            'Excelente': prob_excelente, 'Buena': prob_buena,
            'Regular': prob_regular, 'Mala': prob_mala,
        }
        
        expected_utility = sum(p * self.QUALITY_STATES[state] for state, p in probabilities.items())
        
        return expected_utility, probabilities


class ModelSelectorAgent:
    """
    Agente que selecciona un LLM para maximizar la utilidad esperada del usuario.
    """
    def __init__(self, models_data):
        self.models = models_data
        self.preferences = {}
        self.quality_model = ProbabilisticQualityModel()
        self.higher_is_better_attrs = [
            "context_window", "output_token_per_second", "coding_average",
            "math_average", "data_analysis_average", "science", "writing", "creative_writing"
        ]
        self.lower_is_better_attrs = ["blended_price_per_million_tokens", "first_chunk_latency"]

    @staticmethod
    def _parse_date(date_str):
        try:
            return datetime.strptime(date_str, "%b %Y")
        except ValueError:
            return datetime.min

    def _filter_dominated_models(self, models):
        dominated_indices = set()
        domination_reasons = {}
        n = len(models)
        for i in range(n):
            for j in range(n):
                if i == j or j in dominated_indices: continue
                model_a, model_b = models[i], models[j]
                is_strictly_better_on_one, is_worse_on_any = False, False
                for attr in self.higher_is_better_attrs:
                    if model_a[attr] < model_b[attr]: is_worse_on_any = True; break
                    if model_a[attr] > model_b[attr]: is_strictly_better_on_one = True
                if is_worse_on_any: continue
                for attr in self.lower_is_better_attrs:
                    if model_a[attr] > model_b[attr]: is_worse_on_any = True; break
                    if model_a[attr] < model_b[attr]: is_strictly_better_on_one = True
                if is_worse_on_any: continue
                date_a, date_b = self._parse_date(model_a["knowledge_cutoff"]), self._parse_date(model_b["knowledge_cutoff"])
                if date_a < date_b: is_worse_on_any = True; continue
                if date_a > date_b: is_strictly_better_on_one = True
                if not is_worse_on_any and is_strictly_better_on_one:
                    dominated_indices.add(j)
                    domination_reasons[model_b['model_name']] = model_a['model_name']
        
        report = "### 🔎 Análisis de Dominancia Estricta\n"
        if domination_reasons:
            for dominated, dominator in domination_reasons.items():
                report += f"- **{dominated}** fue descartado (dominado por **{dominator}**).\n"
        else:
            report += "- Ningún modelo fue estrictamente dominado.\n"
            
        filtered_models = [model for i, model in enumerate(models) if i not in dominated_indices]
        return filtered_models, report

    def _normalize_attributes(self, models_to_normalize):
        if not models_to_normalize: return []
        normalized_models = [m.copy() for m in models_to_normalize]
        all_attrs = self.higher_is_better_attrs + self.lower_is_better_attrs
        all_attrs = [attr for attr in all_attrs if attr in normalized_models[0]]
        for attr in all_attrs:
            is_higher_better = attr in self.higher_is_better_attrs
            values = [m[attr] for m in models_to_normalize]
            min_val, max_val = min(values), max(values)
            for model in normalized_models:
                norm_attr_name = f"norm_{attr}"
                if (max_val - min_val) == 0:
                    model[norm_attr_name] = 1.0
                else:
                    norm_val = (model[attr] - min_val) / (max_val - min_val)
                    model[norm_attr_name] = norm_val if is_higher_better else 1 - norm_val
        return normalized_models
    
    def _apply_attitude(self, value, attitude):
        if attitude == 'Averso al Riesgo': return math.sqrt(value)
        elif attitude == 'Buscador de Riesgo': return value ** 2
        else: return value # Neutral al Riesgo

    def _calculate_utility(self, model, category, difficulty, preferences):
        norm_cost = model["norm_blended_price_per_million_tokens"]
        norm_speed = (model["norm_output_token_per_second"] + model["norm_first_chunk_latency"]) / 2
        utility_cost = self._apply_attitude(norm_cost, preferences['cost_attitude'])
        utility_speed = self._apply_attitude(norm_speed, preferences['speed_attitude'])

        eu_quality_score, _ = self.quality_model.get_expected_quality_utility(category, difficulty, model)
        utility_quality = self._apply_attitude(eu_quality_score, preferences['quality_attitude'])

        cost_component = preferences['cost'] * utility_cost
        speed_component = preferences['speed'] * utility_speed
        quality_component = preferences['quality'] * utility_quality
        total_utility = cost_component + speed_component + quality_component

        utility_breakdown = {
            "Costo": cost_component, "Velocidad": speed_component, "Calidad": quality_component
        }
        return total_utility, utility_breakdown

    def select_best_model(self, preferences, category, difficulty):
        total_weight = preferences['cost_weight'] + preferences['speed_weight'] + preferences['quality_weight']
        if total_weight == 0: total_weight = 1

        self.preferences = {
            'cost': preferences['cost_weight'] / total_weight,
            'speed': preferences['speed_weight'] / total_weight,
            'quality': preferences['quality_weight'] / total_weight,
            'cost_attitude': preferences['cost_attitude'],
            'speed_attitude': preferences['speed_attitude'],
            'quality_attitude': preferences['quality_attitude']
        }

        candidate_models, dominance_report = self._filter_dominated_models(self.models)
        if not candidate_models:
            return "No hay modelos candidatos después del filtro de dominancia.", "", "", ""

        normalized_models = self._normalize_attributes(candidate_models)

        for model in normalized_models:
            model['utility'], model['utility_breakdown'] = self._calculate_utility(
                model, category, difficulty, self.preferences
            )

        sorted_models = sorted(normalized_models, key=lambda m: m['utility'], reverse=True)
        best_model = sorted_models[0]

        utility_report = "### 📊 Análisis de Utilidad\n| Modelo | Utilidad Calculada |\n|---|---|\n"
        for model in sorted_models:
            utility_report += f"| {model['model_name']} | {model['utility']:.4f} |\n"
            
        vpi_report = ""
        if len(sorted_models) > 1:
            second_best_model = sorted_models[1]
            utility_diff = best_model['utility'] - second_best_model['utility']
            if utility_diff < (best_model['utility'] * RELATIVE_VPI_THRESHOLD):
                vpi_report = (
                    "### 💡 Análisis del Valor de la Información (VPI)\n"
                    "La decisión es muy reñida (la diferencia de utilidad es menor al 5%). "
                    "Considera obtener más información si la tarea es crítica."
                )
        
        breakdown_report = "### ✅ Recomendación Final\n"
        category_name = [k for k, v in TASK_CATEGORIES.items() if v == category][0]
        breakdown_report += (
            f"Para una tarea de **{category_name}** con dificultad **{difficulty.capitalize()}**, "
            f"el modelo recomendado es **{best_model['model_name']}** "
            f"con una utilidad máxima de **{best_model['utility']:.4f}**.\n\n"
            "**Desglose de la Utilidad:**\n"
        )
        total_utility = sum(best_model['utility_breakdown'].values())
        for component, value in best_model['utility_breakdown'].items():
            percentage = (value / total_utility) * 100 if total_utility > 0 else 0
            breakdown_report += f"- **Aporte de {component}**: {value:.3f} ({percentage:.1f}%)\n"
            
        return dominance_report, utility_report, vpi_report, breakdown_report


def create_interface():
    """
    Función principal que crea y lanza la interfaz de Gradio.
    """
    try:
        models_data = json.load(open('modelos.json', 'r', encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error cargando 'modelos.json': {e}")
        sys.exit(1)

    agent = ModelSelectorAgent(models_data)

    def run_agent_from_interface(
        cost_weight, cost_attitude, speed_weight, speed_attitude, 
        quality_weight, quality_attitude, task_category, task_difficulty, user_prompt
    ):
        if not user_prompt:
            return "Por favor, introduce un prompt.", "", "", ""
            
        preferences = {
            'cost_weight': cost_weight, 'cost_attitude': cost_attitude,
            'speed_weight': speed_weight, 'speed_attitude': speed_attitude,
            'quality_weight': quality_weight, 'quality_attitude': quality_attitude
        }
        category_key = TASK_CATEGORIES[task_category]
        difficulty_key = DIFFICULTY_LEVELS[task_difficulty]
        
        return agent.select_best_model(preferences, category_key, difficulty_key)

    with gr.Blocks(theme=gr.themes.Soft(), title="Agente Selector de LLMs") as iface:
        gr.Markdown("# 🤖 Agente Selector de LLMs")
        gr.Markdown("Esta herramienta te ayuda a elegir el mejor Modelo de Lenguaje Grande (LLM) para tus necesidades específicas, basándose en tus prioridades y la naturaleza de tu tarea.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. Define Tus Prioridades")
                
                gr.Markdown("#### 💰 Prioridad en Costo")
                cost_w = gr.Slider(0, 10, value=5, label="Importancia (0=Baja, 10=Alta)")
                cost_a = gr.Radio(['Averso al Riesgo', 'Neutral al Riesgo', 'Buscador de Riesgo'], value='Averso al Riesgo', label="Actitud frente al riesgo")
                
                gr.Markdown("#### ⚡️ Prioridad en Velocidad")
                speed_w = gr.Slider(0, 10, value=5, label="Importancia (0=Baja, 10=Alta)")
                speed_a = gr.Radio(['Averso al Riesgo', 'Neutral al Riesgo', 'Buscador de Riesgo'], value='Averso al Riesgo', label="Actitud frente al riesgo")

                gr.Markdown("#### 🎯 Prioridad en Calidad")
                quality_w = gr.Slider(0, 10, value=5, label="Importancia (0=Baja, 10=Alta)")
                quality_a = gr.Radio(['Averso al Riesgo', 'Neutral al Riesgo', 'Buscador de Riesgo'], value='Averso al Riesgo', label="Actitud frente al riesgo")
                
                gr.Markdown("## 2. Describe Tu Tarea")
                prompt = gr.Textbox(label="📝 Introduce tu prompt aquí", lines=3, placeholder="Ej: 'Escribe una función en Python que calcule la secuencia de Fibonacci.'")
                task_cat = gr.Dropdown(list(TASK_CATEGORIES.keys()), label="Categoría de la Tarea", value="Programación")
                task_diff = gr.Dropdown(list(DIFFICULTY_LEVELS.keys()), label="Dificultad de la Tarea", value="Media")

                submit_btn = gr.Button("Encontrar el Mejor Modelo", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("## 3. Análisis y Recomendación")
                final_recommendation = gr.Markdown()
                utility_analysis = gr.Markdown()
                dominance_analysis = gr.Markdown()
                vpi_analysis = gr.Markdown()

        inputs = [cost_w, cost_a, speed_w, speed_a, quality_w, quality_a, task_cat, task_diff, prompt]
        outputs = [dominance_analysis, utility_analysis, vpi_analysis, final_recommendation]
        
        submit_btn.click(
            fn=run_agent_from_interface,
            inputs=inputs,
            outputs=outputs
        )

    iface.launch()


if __name__ == "__main__":
    create_interface()