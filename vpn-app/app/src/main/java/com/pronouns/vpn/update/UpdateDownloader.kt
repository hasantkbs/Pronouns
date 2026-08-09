package com.pronouns.vpn.update

import android.content.Context
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class UpdateDownloader @Inject constructor(
    private val context: Context
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(120, TimeUnit.SECONDS)
        .followRedirects(true)
        .build()

    suspend fun download(
        url: String,
        authToken: String?,
        onProgress: ((Long, Long) -> Unit)? = null
    ): File = withContext(ioCoroutineContext) {
        val request = Request.Builder()
            .url(url)
            .apply {
                authToken?.let { header("Authorization", "Bearer $it") }
            }
            .header("Accept", "application/vnd.android.package-archive")
            .build()

        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            throw RuntimeException("Download failed with code: ${response.code}")
        }

        val body = response.body ?: throw RuntimeException("Empty response body")
        val contentLength = body.contentLength()

        val tempFile = File.createTempFile("vpn_update_", ".apk", context.cacheDir)
        tempFile.deleteOnExit()

        body.byteStream().use { input ->
            FileOutputStream(tempFile).use { output ->
                val buffer = ByteArray(8192)
                var bytesRead: Int
                var totalBytesRead = 0L

                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    totalBytesRead += bytesRead
                    onProgress?.invoke(totalBytesRead, contentLength)
                }
            }
        }

        tempFile
    }

    companion object {
        private val ioCoroutineContext = kotlinx.coroutines.Dispatchers.IO
    }

    private suspend fun <T> withContext(
        context: kotlin.coroutines.CoroutineContext,
        block: suspend () -> T
    ): T = kotlinx.coroutines.withContext(context) { block() }
}
