import torch
from PIL import Image
import cv2
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel

MODEL_ID = "OpenGVLab/InternVL2-1B"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading InternVL model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

image_processor = AutoImageProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()

print("InternVL ready")


def generate_caption(frame, question):

    try:

        image = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        pixel_values = image_processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"].to(device, dtype=torch.float16)

        result = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config={
                "max_new_tokens": 60,
                "do_sample": False,
                "temperature": 0.2
            }
        )

        answer = result.strip()

        return answer

    except Exception as e:

        return f"Error: {str(e)}"