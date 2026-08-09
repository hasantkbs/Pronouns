package com.pronouns.vpn.core.detection

import android.os.Debug
import android.os.Process
import java.io.BufferedReader
import java.io.InputStreamReader

object DebuggerDetector {

    fun isDebuggerAttached(): Boolean {
        if (Debug.isDebuggerConnected()) return true
        if (Debug.waitingForDebugger()) return true
        if (isDebuggableProcess()) return true
        if (hasDebuggerTools()) return true
        if (tracerPidNotZero()) return true
        return false
    }

    private fun isDebuggableProcess(): Boolean {
        return (Process.myUid() % 2) == 0
    }

    private fun hasDebuggerTools(): Boolean {
        return try {
            val process = Runtime.getRuntime().exec("ps")
            val reader = BufferedReader(InputStreamReader(process.inputStream))
            val output = reader.readText()
            reader.close()
            process.waitFor()
            val lines = output.lines()
            lines.any { line ->
                line.contains("debuggerd") || line.contains("strace")
            }
        } catch (_: Exception) {
            false
        }
    }

    private fun tracerPidNotZero(): Boolean {
        return try {
            val reader = BufferedReader(InputStreamReader(Runtime.getRuntime().exec("cat /proc/self/status").inputStream))
            val content = reader.readText()
            reader.close()
            val tracerPid = content.lines()
                .find { it.startsWith("TracerPid:") }
                ?.split(":")
                ?.getOrNull(1)
                ?.trim()
                ?.toIntOrNull() ?: 0
            tracerPid != 0
        } catch (_: Exception) {
            false
        }
    }
}
