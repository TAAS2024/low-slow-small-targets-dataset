"""
快速 Depth Demo — 真实模型推理 vs 模拟器对照
=============================================
展示: 真实 Transformer B depth head 输出质量
Seg 用模拟器替代（模型 seg head 因 GT 全0已崩塌）
"""
import json, sys, time, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from transformer_b import TransformerB
from transformers import CLIPModel, CLIPProcessor
from demo import EmbeddingSimulator

SEG_COLORS = {0:(0,0,0),1:(135,206,235),2:(128,128,128),3:(34,139,34),
              4:(210,180,140),5:(30,144,255),6:(255,50,50),7:(255,255,0)}

CASES = [
    ("悬停侦察·城市仰拍", {
        "drone_type":"quadrotor","trajectory":[{"t":0,"action":"hover","distance":50,"norm_u":0,"norm_v":0}],
        "time_of_day":"afternoon","weather":"clear","scene_type":"urban",
        "scene_description":"City skyline, DJI Air 2S hovering at 50m.","modality":"RGB",
        "camera":{"position":"bottom","elevation_deg":30,"fov_deg":60}}),
    ("俯冲接近·工业区俯拍", {
        "drone_type":"quadrotor","trajectory":[{"t":0,"action":"approach","distance":30,"norm_u":0.1,"norm_v":-0.3}],
        "time_of_day":"dusk","weather":"overcast","scene_type":"industrial",
        "scene_description":"Industrial zone, DJI Mavic 3 diving.","modality":"RGB",
        "camera":{"position":"top","elevation_deg":-15,"fov_deg":45}}),
    ("森林巡航·平拍", {
        "drone_type":"quadrotor","trajectory":[{"t":0,"action":"cruise","distance":60,"norm_u":0,"norm_v":0.05}],
        "time_of_day":"morning","weather":"clear","scene_type":"forest",
        "scene_description":"Dense forest, Mini 2 cruising.","modality":"RGB",
        "camera":{"position":"front","elevation_deg":5,"fov_deg":60}}),
]

def arr2img(d): d=(d-d.min())/(d.max()-d.min()+1e-8); return Image.fromarray((d*255).astype(np.uint8),"L").convert("RGB")
def seg2img(s): 
    c=np.zeros((*s.shape,3),np.uint8)
    for k,v in SEG_COLORS.items(): c[s==k]=v
    return Image.fromarray(c)

def label(img, text, color=(255,255,255)):
    d=ImageDraw.Draw(img)
    try: f=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",13)
    except: f=ImageFont.load_default()
    d.rectangle([0,0,img.width,26],fill=(0,0,0,180))
    d.text((5,4),text,fill=color,font=f)
    return img

print("Loading models...")
device="cuda"
clip=CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
proc=CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

ckpt=torch.load("output/layer2/layer2_best.pt",map_location=device,weights_only=False)
model=TransformerB(ckpt["config"]).to(device)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()
sim=EmbeddingSimulator(seed=42)

out=Path("demo_real_output")
out.mkdir(exist_ok=True)

for name,case in CASES:
    t0=time.time()
    j=json.dumps(case,ensure_ascii=False)
    depth_real,_=model.generate([j],clip,proc)
    ms=(time.time()-t0)*1000
    
    d_real=depth_real[0,0].cpu().numpy()
    d_sim=sim.generate_depth_map(case,768)
    s_sim=sim.generate_seg_map(case,768)
    
    # 拼接: 真实depth | 模拟depth | 模拟seg (三栏)
    w=768
    canvas=Image.new("RGB",(w*3,w+26),(30,30,30))
    canvas.paste(label(arr2img(d_real),f"REAL Depth [{d_real.min():.2f},{d_real.max():.2f}] μ={d_real.mean():.2f}",(100,255,100)),(0,0))
    canvas.paste(label(arr2img(d_sim),f"SIM Depth",(255,200,100)),(w,0))
    canvas.paste(label(seg2img(s_sim),f"SIM Seg (model seg broken)",(100,200,255)),(w*2,0))
    
    fn=out/f"depth_demo_{name.split('·')[0]}.png"
    canvas.save(fn)
    print(f"  {name}: {ms:.0f}ms → {fn}")

# 总结
print(f"\nDone. Files in {out.resolve()}/")
for f in sorted(out.glob("depth_demo_*")):
    print(f"  {f.name}  ({f.stat().st_size//1024}KB)")
print("\n📋 3-column layout: REAL Depth | SIM Depth | SIM Seg")
print("Goal: verify depth head produces meaningful spatial structure from JSON text")
