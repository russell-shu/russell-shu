import os
import uuid
from contextlib import nullcontext
import time
from typing import Generator
import sys
sys.path.append('third_party/Matcha-TTS')
import torch.nn
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
from tqdm import tqdm

from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.class_utils import get_model_type
from cosyvoice.utils.file_utils import logging, load_wav


class CosyVoice:

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False):
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f)
        assert get_model_type(configs) != CosyVoice2Model, 'do not use {} for CosyVoice initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoice2Model(configs['llm'],fp16=True)
        self.model.load('{}/llm.pt'.format(model_dir))
        del configs
    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        time_prepare_1 = time.time()
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            counter = 0
            logging.info('synthesis text {}'.format(i))
            print('prepare model input cost: ',time.time()-time_prepare_1)
            # time_prepare_1 = time.time()
            for model_output in self.model.tts(**model_input):
                # speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                # logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                counter+=1
                if counter % 15 == 0:
                    print('耗时',time.time()-start_time)
                    start_time = time.time()
            time_prepare_1 = time.time()
class CosyVoice2(CosyVoice):

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False):

        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        # assert get_model_type(configs) == CosyVoice2Model, 'do not use {} for CosyVoice2 initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoice2Model(configs['llm'],fp16)
        self.model.load('{}/llm.pt'.format(model_dir))

        del configs


class CosyVoice2Model:
    def __init__(self,
                 llm:torch.nn.Module,
                 fp16:bool):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.tts_speech_token_dict = {}
        self.fp16 = fp16
        self.llm.fp16 = fp16
        self.llm_end_dict = {}
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        self.llm_context = torch.cuda.stream(
            torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()

        if self.fp16 is True:
            self.llm.half()


    def load(self,llm_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device), strict=True)
        self.llm.to(self.device).eval()

        # 验证所有参数的 dtype 是否匹配 fp16 设置
        first_param = next(self.llm.parameters())
        print(f"Parameter dtype: {first_param.dtype}")
        if self.fp16:
                for name, param in self.llm.named_parameters():
                    if param.dtype != torch.float16:
                        print(f"[ERROR] Parameter {name} is {param.dtype}, expected torch.float16")
        else:
            for name, param in self.llm.named_parameters():
                if param.dtype != torch.float32:
                    print(f"[ERROR] Parameter {name} is {param.dtype}, expected torch.float32")


    def tts(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())

        self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False

        for i in self.llm_job(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid):
            yield i
    def llm_job(self,text,prompt_text,llm_prompt_speech_token,llm_embedding,uuid):
        for i in self.llm.inference(text=text.to(self.device),
                                    text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                    prompt_text=prompt_text.to(self.device),
                                    prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                    prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                    prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]],
                                                                         dtype=torch.int32).to(self.device),
                                    embedding=llm_embedding.to(self.device)):
            # print(i)
            yield i


cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, fp16=True)
text = '为啥我这个没有捕获到.嗨，我是通义灵码，你的智能编码助手。你可以问我编码相关的问题或提交反馈让我变得更好。如果需要了解更多我的能力、动态及企业版信息，可以前往查看帮助文档。更详细的诉求描述和更多的上下文（如文件、图片、提交等），可以让我更加理解你的诉求，并给出更好的方案和代码建议。同时，智能体模式时，我可以使用很多工具来解决复杂的编码任务，你也可以为我添加更多贴合你工作流程的 MCP 工具。装环境需要那边提供服务器的硬件型号，你那边有负责提供硬件的人能问到吗？顺利的话两三天能把昇腾提供的Cosyvoice的demo跑起来，然后再花个三四天基于这个demo开发服务接入咱们系统'
prompt_text = '你好，你好啊我是智能客服灵犀,能查话费,流量,账单,套餐,有啥问题都可以问我。'
prompt_speech_16k = load_wav('./asset/xiaolv_fast_v2.wav', 16000)
for i in cosyvoice.inference_zero_shot(text, prompt_text,prompt_speech_16k, stream=False):
    pass