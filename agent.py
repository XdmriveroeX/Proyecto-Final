import sys
import json
import math
from datetime import datetime

# --- Constantes de Configuración ---

# Diccionario que mapea la elección del usuario a la clave de atributo del modelo
TASK_CATEGORIES = {
    "1": "coding_average",
    "2": "math_average",
    "3": "data_analysis_average",
    "4": "science",
    "5": "writing",
    "6": "creative_writing",
}

# MEJORA: El umbral para el análisis VPI ahora es relativo (5%).
# Si la diferencia de utilidad entre los dos mejores modelos es menor al 5%
# de la utilidad del mejor, la decisión se considera "sensible".
RELATIVE_VPI_THRESHOLD = 0.05

def load_models_from_json(filepath="modelos.json"):
    """
    Carga los datos de los modelos desde un archivo JSON.
    Maneja errores si el archivo no se encuentra o tiene un formato incorrecto.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            models_data = json.load(f)
        return models_data
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{filepath}'.")
        print("Asegúrate de que el archivo exista en el mismo directorio que el script.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: El archivo '{filepath}' no es un JSON válido.")
        print("Por favor, revisa la sintaxis del archivo.")
        sys.exit(1)


class ModelSelectorAgent:
    """
    Un agente que selecciona el mejor LLM aplicando el Principio de Máxima Utilidad Esperada,
    filtrando por dominancia estricta, modelando la actitud ante el riesgo y razonando sobre el Valor de la Información.
    """
    def __init__(self, models_data):
        """
        Inicializa el agente con los datos de los modelos disponibles.
        """
        self.models = models_data
        self.preferences = {}
        # Atributos para la comprobación de dominancia
        self.higher_is_better_attrs = [
            "context_window", "output_token_per_second", "coding_average", 
            "math_average", "data_analysis_average", "science", "writing", "creative_writing"
        ]
        self.lower_is_better_attrs = ["blended_price_per_million_tokens", "first_chunk_latency"]

    @staticmethod
    def _parse_date(date_str):
        """Parsea una fecha en formato 'Mes Año' a un objeto datetime."""
        try:
            return datetime.strptime(date_str, "%b %Y")
        except ValueError:
            # Devuelve una fecha muy antigua si el formato es inesperado
            return datetime.min

    def _filter_dominated_models(self, models):
        """
        Implementa el filtro de Dominancia Estricta.
        Un modelo B es eliminado si existe un modelo A que es igual o mejor en todos los atributos
        y estrictamente mejor en al menos un atributo.
        """
        dominated_indices = set()
        n = len(models)

        for i in range(n):
            for j in range(n):
                if i == j or j in dominated_indices:
                    continue

                # model_a es el potencial dominador, model_b es el potencial dominado
                model_a = models[i]
                model_b = models[j]

                is_strictly_better_on_one = False
                is_worse_on_any = False

                # Comprobar atributos donde más es mejor
                for attr in self.higher_is_better_attrs:
                    if model_a[attr] < model_b[attr]:
                        is_worse_on_any = True; break
                    if model_a[attr] > model_b[attr]:
                        is_strictly_better_on_one = True
                if is_worse_on_any: continue

                # Comprobar atributos donde menos es mejor
                for attr in self.lower_is_better_attrs:
                    if model_a[attr] > model_b[attr]:
                        is_worse_on_any = True; break
                    if model_a[attr] < model_b[attr]:
                        is_strictly_better_on_one = True
                if is_worse_on_any: continue

                # Comprobar la fecha de corte del conocimiento
                date_a = self._parse_date(model_a["knowledge_cutoff"])
                date_b = self._parse_date(model_b["knowledge_cutoff"])
                if date_a < date_b: is_worse_on_any = True; continue
                if date_a > date_b: is_strictly_better_on_one = True

                # Conclusión: si A nunca es peor y es mejor en al menos un aspecto, A domina a B
                if not is_worse_on_any and is_strictly_better_on_one:
                    dominated_indices.add(j)

        if dominated_indices:
            print("\n--- 🔎 FILTRO DE DOMINANCIA ESTRICTA ---")
            dominated_names = [models[i]['model_name'] for i in sorted(list(dominated_indices))]
            print(f"Modelos eliminados por ser inferiores: {', '.join(dominated_names)}")
        
        return [model for i, model in enumerate(models) if i not in dominated_indices]

    def _normalize_attributes(self, models_to_normalize):
        """
        Normaliza los atributos de una lista de modelos a una escala de 0 a 1.
        """
        if not models_to_normalize: return []
        normalized_models = [m.copy() for m in models_to_normalize]
        
        all_attrs = ["output_token_per_second"] + list(TASK_CATEGORIES.values()) + ["blended_price_per_million_tokens", "first_chunk_latency"]
        
        for attr in all_attrs:
            is_higher_better = attr not in ["blended_price_per_million_tokens", "first_chunk_latency"]
            values = [m[attr] for m in models_to_normalize]
            min_val, max_val = min(values), max(values)
            
            for model in normalized_models:
                if (max_val - min_val) == 0:
                    model[f"norm_{attr}"] = 1.0
                else:
                    norm_val = (model[attr] - min_val) / (max_val - min_val)
                    model[f"norm_{attr}"] = norm_val if is_higher_better else 1 - norm_val
        
        return normalized_models

    def elicit_preferences(self):
        """
        Implementa la Elicitación de Preferencias, incluyendo la actitud ante el riesgo.
        """
        print("\n--- 📝 DEFINE TUS PRIORIDADES ---")
        print("En una escala de 0 (no importa) a 10 (máxima prioridad), ¿qué tan importante es cada factor?")
        
        weights = {}
        try:
            weights['cost'] = float(input("💰 Prioridad en AHORRAR DINERO (Costo): "))
            risk_choice = input("   => Frente al costo, ¿cuál es tu actitud? [1] Averso, [2] Neutral, [3] Buscador: ")
            if risk_choice not in ["1", "2", "3"]:
                print("   Opción inválida. Se usará 'Neutral' por defecto."); risk_choice = "2"
            
            weights['speed'] = float(input("⚡️ Prioridad en la RAPIDEZ (Velocidad): "))
            weights['quality'] = float(input("🎯 Prioridad en la PRECISIÓN (Calidad): "))
        except ValueError:
            print("\n⚠️ Entrada inválida. Usando pesos y actitud por defecto.")
            self.preferences = {'cost': 1/3, 'speed': 1/3, 'quality': 1/3, 'risk_attitude': '2'}
            return

        total_weight = sum(weights.values())
        if total_weight == 0:
            print("\n⚠️ La suma de pesos no puede ser cero. Usando pesos por defecto.")
            self.preferences = {'cost': 1/3, 'speed': 1/3, 'quality': 1/3, 'risk_attitude': '2'}
        else:
            self.preferences = {k: v / total_weight for k, v in weights.items()}
            self.preferences['risk_attitude'] = risk_choice

        print("\nPreferencias configuradas:")
        for k, v in self.preferences.items():
            if k != 'risk_attitude': print(f"- {k.capitalize()} (Peso): {v:.2f}")
        risk_map = {'1': 'Averso al Riesgo', '2': 'Neutral', '3': 'Buscador de Riesgo'}
        print(f"- Actitud frente al costo: {risk_map[self.preferences['risk_attitude']]}")

    def _calculate_utility(self, model, task_key, preferences):
        """
        Implementa la Función de Utilidad Multiatributo, incorporando la actitud ante el riesgo.
        """
        cost_utility = model["norm_blended_price_per_million_tokens"]
        risk_attitude = preferences.get('risk_attitude', '2')

        if risk_attitude == '1':    # Aversión al Riesgo (Función Cóncava -> U(x) = x^2)
            cost_utility = cost_utility ** 2
        elif risk_attitude == '3':  # Búsqueda de Riesgo (Función Convexa -> U(x) = sqrt(x))
            cost_utility = math.sqrt(cost_utility)

        norm_speed_score = (model["norm_output_token_per_second"] + model["norm_first_chunk_latency"]) / 2
        
        utility = (
            preferences['cost'] * cost_utility +
            preferences['speed'] * norm_speed_score +
            preferences['quality'] * model[f"norm_{task_key}"]
        )
        return utility

    def select_best_model(self, task_key):
        """
        Ejecuta el ciclo de decisión: Filtra, Normaliza, Calcula Utilidad y Analiza.
        """
        if not self.preferences: self.elicit_preferences()

        candidate_models = self._filter_dominated_models(self.models)
        if not candidate_models:
            print("\n❌ No quedaron modelos candidatos tras el filtro de dominancia.")
            return None

        normalized_models = self._normalize_attributes(candidate_models)
        
        for model in normalized_models:
            model['utility'] = self._calculate_utility(model, task_key, self.preferences)
        
        sorted_models = sorted(normalized_models, key=lambda m: m['utility'], reverse=True)
        best_model = sorted_models[0]
        
        print("\n--- 📊 ANÁLISIS DE UTILIDAD ---")
        print(f"{'Modelo':<25} {'Utilidad Calculada'}")
        print("-" * 45)
        for model in sorted_models:
            print(f"{model['model_name']:<25} {model['utility']:.4f}")

        if len(sorted_models) > 1:
            second_best_model = sorted_models[1]
            utility_diff = best_model['utility'] - second_best_model['utility']
            
            # MEJORA: El umbral de VPI ahora es relativo
            if utility_diff < (best_model['utility'] * RELATIVE_VPI_THRESHOLD):
                print("\n--- 💡 ANÁLISIS DEL VALOR DE LA INFORMACIÓN (VPI) ---")
                print("La decisión está muy reñida (diferencia de utilidad menor al 5%).")
                print("El valor de obtener más información (ej. una prueba real) podría ser alto.")
                print(f"Sugerencia: Considera probar los dos mejores modelos:")
                print(f"  1. {best_model['model_name']} (Utilidad: {best_model['utility']:.3f})")
                print(f"  2. {second_best_model['model_name']} (Utilidad: {second_best_model['utility']:.3f})")

        return best_model

def main():
    """
    Función principal que ejecuta la simulación del agente.
    """
    print("🤖 Agente de Selección de Modelos de Lenguaje (v3.0 con Dominancia y VPI Relativo) 🤖")
    
    models_data = load_models_from_json()
    agent = ModelSelectorAgent(models_data)

    while True:
        agent.elicit_preferences()

        print("\n--- 📚 SELECCIÓN DE TAREA ---")
        print("¿Para qué tipo de tarea necesitas el modelo?")
        for key, value in TASK_CATEGORIES.items():
            readable_name = value.replace('_', ' ').replace('average', '').strip().capitalize()
            print(f"  {key}) {readable_name}")
        
        task_choice = input(f"Elige una opción (1-{len(TASK_CATEGORIES)}): ")

        if task_choice not in TASK_CATEGORIES:
            print("Opción inválida. Por favor, intenta de nuevo.")
            continue

        selected_task_key = TASK_CATEGORIES[task_choice]
        
        best_model = agent.select_best_model(selected_task_key)

        if best_model:
            print("\n--- ✅ DECISIÓN FINAL DEL AGENTE ---")
            print(f"Basado en tus preferencias, el modelo recomendado es: **{best_model['model_name']}**")
            print(f"   Calculó una utilidad máxima de: {best_model['utility']:.4f}\n")
        
        again = input("¿Deseas realizar otra consulta? (s/n): ").lower()
        if again != 's':
            print("\n👋 ¡Hasta luego!")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrograma interrumpido. Adiós.")
        sys.exit(0)