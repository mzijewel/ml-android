package dev.jewel.machinelearning

import android.content.ContentResolver
import android.content.Context
import android.database.Cursor
import android.net.Uri
import android.provider.MediaStore

fun Uri.toImagePath(context: Context): String {
    val projection = arrayOf(MediaStore.Images.Media.DATA)
    var imagePath: String =""
    val contentResolver: ContentResolver = context.contentResolver
    val cursor: Cursor? = contentResolver.query(this, projection, null, null, null)
    cursor?.use { cursor ->
        val columnIndex = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA)
        if (cursor.moveToFirst()) {
            imagePath = cursor.getString(columnIndex)
        }
    }
    cursor?.close()
    return imagePath
}