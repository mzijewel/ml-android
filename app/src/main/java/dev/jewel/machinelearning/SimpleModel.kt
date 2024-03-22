package dev.jewel.machinelearning

import android.content.Context
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor

/* simple equation
    y=3x+2
 */
class SimpleModel {
    fun run(context: Context){
        // Load the PyTorch model
        val module = Module.load(Utils.assetFilePath(context, "m1.pt"))

        val input= floatArrayOf(2f)
        val shape= longArrayOf(input.size.toLong())

        val inputTensor = Tensor.fromBlob(input,shape)


        // Run inference
        val outputTensor: IValue = module.forward(IValue.from(inputTensor))

        // Get output tensor as float array
        val outputArray = outputTensor.toTensor().dataAsFloatArray

        for (value in outputArray) {
            Log.d("PyTorch Output", value.toString())
        }
    }
    fun run2(context: Context){
        // python: SimpleEqnTorch.py
        // y=2a+3b+4c
        val modelName="m_eqn.pt"
        // Load the PyTorch model
        val module = Module.load(Utils.assetFilePath(context, modelName))

        val input= floatArrayOf(1f,2f,3f)
        val shape= longArrayOf(input.size.toLong())

        val inputTensor = Tensor.fromBlob(input,shape)


        // Run inference
        val outputTensor: IValue = module.forward(IValue.from(inputTensor))

        // Get output tensor as float array
        val outputArray = outputTensor.toTensor().dataAsFloatArray

        for (value in outputArray) {
            Log.d("PyTorch Output", value.toString())
        }
    }
}