import pyaudiowpatch as pyaudio
p = pyaudio.PyAudio()
print(p.get_host_api_info_by_type(pyaudio.paWASAPI))
print("---loopback devices---")
for dev in p.get_loopback_device_info_generator():
    print(dev)