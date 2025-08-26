from google import genai
import random
import time

def concepts_init(client):
    """
    Function to initialize key concepts for concept evolution. The initial concept set is pre-generated and we keep both
        the prompt and its corresponding model response as historical records.
    
    Returns:
        List of key concepts.
    """

    history = []
    history.append({
        "role": "user",
        "parts": [
                    {
                    "text": "You are an expert in multimodal interaction analysis. Multimodal intent understanding aims to capture \
                    the speaker's intent based on the video and the spoken text. The possible intent list is ['Complain', 'Praise', \
                    'Apologise', 'Thank', 'Criticize', 'Agree', 'Taunt', 'Flaunt', 'Joke', 'Oppose', 'Comfort', 'Care', 'Inform', \
                    'Advise', 'Arrange', 'Introduce', 'Leave', 'Prevent', 'Greet', 'Ask for help']. \
                    Take the video with text (so your new desk is gonna be here in three days.) as an example, the intent can be \
                    inferred from various key concepts such as facial expressions, body movements, tone of voice, volume, specific \
                    intent words, and so on. Please analyze what key concepts are essential to the multimodal intent recognition task, \
                    and list the top 30 most important concepts that would help in understanding speaker intent across different scenarios.\
                    The concepts should be semantically meaningful, discriminative, and broadly applicable across multiple intent categories.\
                    Avoid concepts that simply duplicate intent labels. Respond only with the list of key concepts, separated by commas."
                    }
                ]
    })

    response = "Facial Expression, Tone of Voice, Body Language, Gaze Direction, Word Choice, Sentence Structure, Contextual Clues, Speaker's Role, \
                Addressee's Role, Relationship Dynamics, Politeness Markers, Intensity, Formality, Directness, Emotional State, Certainty, Hesitation, \
                Humor, Irony, Sarcasm, Emphasis, Topic of Discussion, Prior Interactions, Cultural Norms, Background Noise, Visual Cues, Pauses, Interruptions, \
                Speech Rate, Volume."
    history.append({
        "role": "model",
        "parts": [{"text": response}]
    })
    key_concepts = [c.strip().lower() for c in response.rstrip('.').split(',') if c.strip()]
    return key_concepts, history


def concepts_evolution(client, history, concept, D_score, S_score, epoch_num, loss, eval_score):
    """
    Function to perform concept evolution in a multimodal intent model.
    
    Args:
        model: The multimodal intent model.
        data_loader: DataLoader for the training data.
        optimizer: Optimizer for updating model parameters.
        criterion: Loss function.
        device: Device to run the model on (CPU or GPU).
    
    Returns:
        None
    """
    concept_lines = [f"{name}: D_score = {d_score:.4f}, S_score = {s_score:.4f}" for name, d_score, s_score in zip(concept, D_score, S_score)]
    concept_info = "\n".join(concept_lines)


    history.append({
        "role": "user",
        "parts": [
                {
                "text": f"""
                    You are an assistant for training a multimodal intent recognition model that relies on a set of semantic concepts to enhance interpretability and performance.

                    At this stage, you are provided with:
                    - The current list of concepts, their associated discriminability scores (D_score), where **higher D_score indicates stronger discriminative ability** across intent categories, and their similarity scores (S_score), where higher values indicate stronger semantic overlap with other concepts.
                    - The current epoch number, training loss, and validation performance score.
                    - The goal is to iteratively refine the concept list by removing weak or redundant concepts and introducing more effective ones.

                    **Instructions:**
                    1. Review the concepts and their D_scores.
                    2. Identify the **top 5 most discriminative concepts** (highest D_scores) and analyze their common semantic or structural characteristics.
                    3. Use these shared characteristics to **propose new concepts** that aligns well with them.
                    4. Replace the **least discriminative concept** (least D_score) and  with new concept.
                    5. Replace the **highest similar concept** (highest S_score) and  with new concept.
                    6. Ensure the total number of concepts remains **unchanged**.

                    **Training Status:**
                    - Epoch: {epoch_num:.4f}
                    - Training Loss: {loss:.4f}
                    - Validation Score: {eval_score:.4f}

                    **Concepts and Scores:**
                    {concept_info}

                    **Output Format:**
                    Return **only** the updated concept list as a comma-separated string. **Do not include explanations or rankings.**
                    """
                }
            ]
    })
    response = safe_generate_content(
        client=client,
        model='models/gemini-2.0-flash',
        history=history
    )
    if response is None:
        print("[INFO] Concept list unchanged due to API failure.")
        return concept, history
    
    history.append({
        "role": "model",
        "parts": [{"text": response.text}]
    })
    key_concepts = [c.strip().lower() for c in response.text.rstrip('.').split(',') if c.strip()]
    return key_concepts, history, concept_info



def safe_generate_content(client, model, history, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=history
            )
            return response
        except Exception as e:
            # If it is a temporary error, try again
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"[Retrying in {sleep_time:.2f}s] Model busy or unavailable, attempt {attempt + 1}/{max_retries}")
                time.sleep(sleep_time)
            else:
                print(f"[ERROR] Non-retryable error encountered: {e}")
                return None
    print(f"[WARNING] Failed after {max_retries} retries. Keeping previous concept list.")
    return None