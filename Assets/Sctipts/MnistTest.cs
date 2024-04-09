using System.Linq;
using UnityEngine;
using Unity.Barracuda;
using UI = UnityEngine.UI;
using System.Runtime.InteropServices;

sealed class MnistTest : MonoBehaviour
{
    public NNModel _model;
    public Texture2D _image;
    public UI.RawImage _imageView;
    public UI.Text _textView;

    void Start()
    {
        // Convert the input image into a 1x28x28x1 tensor.
        using Tensor input = new Tensor(1, 28, 28, 1);

        for (int y = 0; y < 28; y++)
        {
            for (int x = 0; x < 28; x++)
            {
                int tx = x * _image.width / 28;
                int ty = y * _image.height / 28;
                input[0, 27 - y, x, 0] = _image.GetPixel(tx, ty).grayscale;
            }
        }



        // Run the MNIST model.
        using IWorker worker =
          ModelLoader.Load(_model).CreateWorker(WorkerFactory.Device.CPU);

        worker.Execute(input);


        // Inspect the output tensor.
        Tensor output = worker.PeekOutput();

        float[] scores = Enumerable.Range(0, 10).
                     Select(i => output[0, 0, 0, i]).SoftMax().ToArray();

        
        // ƒƒO‚É•\Ž¦
        foreach (float a in scores) 
        {
            Debug.Log($"{a}");
        }


        // Show the results on the UI.
        _imageView.texture = _image;
        _textView.text = Enumerable.Range(0, 10).
                         Select(i => $"{i}: {scores[i]:0.00}").
                         Aggregate((t, s) => t + "\n" + s);
    }
}
