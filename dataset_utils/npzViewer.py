import numpy as np

if __name__ == "__main__":
    data = np.load('D:\project_summary\Graduation Project\\tmp\datasets_np\\flag_simple\\train\\ex0.npz',allow_pickle=True)
    for a, b in data.items():
        print("key",a)
        print("value",b[0])
        