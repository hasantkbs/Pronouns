package com.pronouns.vpn.security

import android.os.Debug
import java.io.BufferedReader
import java.io.FileReader
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class DebuggerDetector @Inject constructor() {

    fun isDebuggerAttached(): Boolean {
        return checkDebuggerFlag() ||
                checkTracerPid() ||
                checkWaitForDebugger() ||
                checkIsDebug()
    }

    private fun checkDebuggerFlag(): Boolean {
        return Debug.isDebuggerConnected()
    }

    private fun checkTracerPid(): Boolean {
        return try {
            BufferedReader(FileReader("/proc/self/status")).use { reader ->
                var line: String?
                while (reader.readLine().also { line = it } != null) {
                    if (line!!.startsWith("TracerPid:")) {
                        val pid = line!!.split(":").getOrNull(1)?.trim() ?: "0"
                        return pid != "0"
                    }
                }
            }
            false
        } catch (e: Exception) {
            false
        }
    }

    private fun checkWaitForDebugger(): Boolean {
        return Debug.waitingForDebugger()
    }

    private fun checkIsDebug(): Boolean {
        return try {
            val process = Runtime.getRuntime().exec(
                arrayOf("getprop", "ro.debuggable")
            )
            val reader = java.io.BufferedReader(
                java.io.InputStreamReader(process.inputStream)
            )
            val value = reader.readLine()
            process.destroy()
            value == "1"
        } catch (e: Exception) {
            false
        }
    }
}
