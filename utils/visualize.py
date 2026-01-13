import matplotlib.pyplot as plt
from typing import List, Optional

def viz(train_hist: List[float], val_hist: List[float], title: Optional[str]):
    plt.figure()
    if title is None:
       plt.title("Grafik visualisasi training") 
    plt.title("Perbandingan train model dan val model")
    plt.plot(train_hist, c='r')
    plt.plot(val_hist, c='b')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True)
    plt.show()