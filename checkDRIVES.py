import torch
def check_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print({torch.cuda.is_available()})
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("GPU", {"name": props.name, "VRAM_GB": round(props.total_memory/1e9,2)})
        if props.total_memory < 7e9:
           print("⚠️  <8GB VRAM — reduce batch size in config.py!", "⚠️")
    else:
        print("No GPU found — CPU training will be very slow!", "⚠️")
    return device

check_gpu()