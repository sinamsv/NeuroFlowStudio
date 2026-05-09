# 🧠✨ NeuroFlow Studio

> ⚠️ MVP Status
This is an MVP focused on real-time performance and quick iteration. The code is intentionally single-file and optimized for low latency between PyTorch and Pygame. Expect rapid changes and occasional rough edges; refactors and multi-file structure are planned for future versions.


Welcome to **NeuroFlow Studio**! 👋 

If you've ever wanted to visually experiment with artificial intelligence and see how different neural network architectures perform, you're in the right place. Think of it as a local, high-performance version of TensorFlow Playground, but completely powered by **PyTorch**! 🔥

### 🤔 What is this project all about?
I built NeuroFlow Studio primarily as an educational tool. It's a simulation environment that lets you test out various AI models right on your own computer. You can tweak architectures, play around with complex structures, and visually discover which setups work best for different tasks. 

To keep things running buttery smooth, the UI is built entirely with **Pygame** 🎮, while **NumPy** handles the heavy mathematical calculations behind the scenes. This means no web-based latency, smooth real-time visual feedback, and top-notch performance. 

### ✨ Key Features
Here are some of the cool things you can do in the studio:
*   🎨 **Manual Dataset Creation:** You can actually draw and create your own point-based datasets manually right on the screen! It's a super fun way to see how models react to custom data shapes.
*   ⚙️ **Total Layer Flexibility:** You have full control over the network's architecture. You can easily adjust the number of neurons in each individual layer and swap out activation functions to see what works best.

### 🚀 How to get it running
Setup is super easy! Just open up your terminal and follow these quick steps:


**1. Clone the repository:**
```bash
git clone https://github.com/sinamsv/NeuroFlowStudio.git
cd NeuroFlowStudio
```

**2. Install the dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the studio:**
```bash
python3 main.py
```
And that's it! You're ready to start simulating. 🎉

### 🛠️ A quick note for developers
To squeeze out the absolute best performance and keep latency to a bare minimum, I've kept the entire project inside a single file. I know this might make reading or editing the code a little tricky if you're diving into the source, so thanks for bearing with me on that design choice! 😊

### 🐛 Found a bug?
Since this is a growing project, you might run into a hiccup here and there. If you find any bugs or have ideas for improvements, I would absolutely love it if you could open an **Issue**. Let's make this tool better together! 

Happy experimenting! 🌌🤖