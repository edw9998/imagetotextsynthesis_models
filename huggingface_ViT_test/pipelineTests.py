from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

# implemented pipeline = https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
model = VisionEncoderDecoderModel.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
feature_extractor = ViTImageProcessor.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
tokenizer = AutoTokenizer.from_pretrained('nlpconnect/vit-gpt2-image-captioning')

# force use cuda tensor/host GPU and add this pipeline to host device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# hyperparameters
max_length = 16
num_beams = 4
gen_kwargs = {'max_length': max_length, 'num_beams': num_beams}

class huggingFace_ViT_GPT2:
    @staticmethod
    def generate_caption(f_paths):
        images = []
        for f_path in f_paths:
            image_val = Image.open(f_path)
            # convert to RGB(a)
            if(image_val.mode != 'RGB'):
                image_val = image_val.convert(mode = 'RGB')
            images.append(image_val)
        
        pixel_values = feature_extractor(images = images, return_tensors = 'pt').pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = model.generate(pixel_values, **gen_kwargs)

        preds = tokenizer.batch_decode(output_ids, skip_special_tokens = True)
        # for the current instance only generate 1 candidate result for comparison with reference captions.
        preds = [pred.strip() for pred in preds][0]

        print("startseq" + str(preds) + "endseq")
        return str(preds)

test = huggingFace_ViT_GPT2()

# response = startseq['a statue of a penguin with a face painted on it']endseq
# test.generate_caption(['f_paths\\test_1.jpeg'])

# response = startseq['a man holding a cell phone in his hand']endseq
# test.generate_caption(['f_paths\\test_2.jpg'])