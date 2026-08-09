package com.pronouns.vpn.security

import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class FridaDetector @Inject constructor() {

    private val fridaLibraries = listOf(
        "frida-agent.so",
        "frida-gadget.so",
        "frida-gadget",
        "frida-helper"
    )

    private val fridaKnownPaths = listOf(
        "/data/local/tmp/frida-server",
        "/data/local/tmp/re.frida.server",
        "/data/local/tmp/frida",
        "/sdcard/frida"
    )

    private val fridaPorts = listOf(
        27042, // default
        27043, // main
        27044, // trace
        27045  // trace
    )

    fun isFridaPresent(): Boolean {
        return checkFridaLibraries() ||
                checkFridaProcesses() ||
                checkFridaPaths() ||
                checkFridaPorts() ||
                checkFridaThreads() ||
                checkDbusResponse()
    }

    private fun checkFridaLibraries(): Boolean {
        return try {
            val maps = File("/proc/self/maps")
            if (maps.exists()) {
                val content = maps.readText()
                fridaLibraries.any { content.contains(it, ignoreCase = true) }
            } else false
        } catch (e: Exception) {
            false
        }
    }

    private fun checkFridaProcesses(): Boolean {
        return try {
            val process = Runtime.getRuntime().exec("ps")
            val reader = BufferedReader(InputStreamReader(process.inputStream))
            var line: String?
            while (reader.readLine().also { line = it } != null) {
                if (line!!.contains("frida", ignoreCase = true)) return true
            }
            process.destroy()
            false
        } catch (e: Exception) {
            false
        }
    }

    private fun checkFridaPaths(): Boolean {
        return fridaKnownPaths.any { File(it).exists() }
    }

    private fun checkFridaPorts(): Boolean {
        return try {
            for (port in fridaPorts) {
                try {
                    val socket = java.net.Socket()
                    socket.connect(java.net.InetSocketAddress("127.0.0.1", port), 100)
                    socket.close()
                    return true
                } catch (e: java.net.ConnectException) {
                    continue
                }
            }
            false
        } catch (e: Exception) {
            false
        }
    }

    private fun checkFridaThreads(): Boolean {
        return try {
            val threadsDir = File("/proc/self/task")
            if (threadsDir.exists()) {
                val threads = threadsDir.listFiles() ?: return false
                for (thread in threads) {
                    try {
                        val status = File(thread, "status")
                        if (status.exists()) {
                            val content = status.readText()
                            if (content.contains("gmain") ||
                                content.contains("gdbus") ||
                                content.contains("gum-js-loop")
                            ) return true
                        }
                    } catch (e: Exception) {
                        continue
                    }
                }
            }
            false
        } catch (e: Exception) {
            false
        }
    }

    private fun checkDbusResponse(): Boolean {
        return try {
            val socket = java.net.Socket()
            socket.connect(java.net.InetSocketAddress("127.0.0.1", 27042), 200)
            val header = byteArrayOf(
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x01.toByte(), 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
            )
            socket.getOutputStream().write(header)
            val response = ByteArray(16)
            val bytesRead = socket.getInputStream().read(response)
            socket.close()

            bytesRead >= 16 && response[12] == 'R'.code.toByte() &&
                    response[13] == 'E'.code.toByte() &&
                    response[14] == 'J'.code.toByte()
        } catch (e: Exception) {
            false
        }
    }
}
