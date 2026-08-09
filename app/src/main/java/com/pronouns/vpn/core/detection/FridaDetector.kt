package com.pronouns.vpn.core.detection

import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import java.net.Socket

object FridaDetector {

    private val fridaBinaryPaths = listOf(
        "/data/local/tmp/frida-server",
        "/data/local/tmp/frida-gadget",
        "/data/local/tmp/re.frida.server",
        "/data/local/tmp/frida-agent-32.so",
        "/data/local/tmp/frida-agent-64.so"
    )

    private val fridaSystemPaths = listOf(
        "/system/bin/frida-server"
    )

    private val fridaPorts = listOf(27042, 27043)

    fun isFridaPresent(): Boolean {
        if (checkFridaBinaries()) return true
        if (checkFridaSystemPaths()) return true
        if (checkFridaInMaps()) return true
        if (checkFridaPorts()) return true
        return false
    }

    private fun checkFridaBinaries(): Boolean {
        return fridaBinaryPaths.any { path -> File(path).exists() }
    }

    private fun checkFridaSystemPaths(): Boolean {
        return fridaSystemPaths.any { path -> File(path).exists() }
    }

    private fun checkFridaInMaps(): Boolean {
        return try {
            val reader = BufferedReader(
                InputStreamReader(Runtime.getRuntime().exec("cat /proc/self/maps").inputStream)
            )
            val content = reader.readText()
            reader.close()
            val lines = content.lines()
            lines.any { line ->
                line.contains("libfrida") || line.contains("frida-agent") || line.contains("frida-gadget")
            }
        } catch (_: Exception) {
            false
        }
    }

    private fun checkFridaPorts(): Boolean {
        return fridaPorts.any { port ->
            try {
                val socket = Socket("127.0.0.1", port)
                socket.close()
                true
            } catch (_: Exception) {
                false
            }
        }
    }
}
