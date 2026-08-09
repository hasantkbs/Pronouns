package com.pronouns.vpn.security

import android.os.Build
import java.io.File
import java.io.FileInputStream
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class EmulatorDetector @Inject constructor() {

    private val knownEmulatorProperties = listOf(
        "ro.kernel.qemu",
        "ro.secure",
        "ro.debuggable",
        "ro.product.cpu.abi2"
    )

    private val knownEmulatorFiles = listOf(
        "/system/lib/libc_malloc_debug_qemu.so",
        "/system/bin/qemu-props",
        "/system/bin/qemu-trace",
        "/dev/socket/qemud",
        "/dev/qemu_pipe"
    )

    fun isEmulator(): Boolean {
        return checkBuildProperties() ||
                checkEmulatorFiles() ||
                checkPhoneNumber() ||
                checkEmulatorDrivers() ||
                checkQemuProps() ||
                checkGenymotion()
    }

    private fun checkBuildProperties(): Boolean {
        val device = Build.DEVICE?.lowercase() ?: ""
        val model = Build.MODEL?.lowercase() ?: ""
        val manufacturer = Build.MANUFACTURER?.lowercase() ?: ""
        val product = Build.PRODUCT?.lowercase() ?: ""
        val hardware = Build.HARDWARE?.lowercase() ?: ""
        val fingerprint = Build.FINGERPRINT?.lowercase() ?: ""

        return device.startsWith("generic") ||
                device.contains("emulator") ||
                device.contains("android_x86") ||
                device.contains("android_x86_64") ||
                model.contains("emulator") ||
                model.contains("sdk") ||
                manufacturer.contains("genymotion") ||
                manufacturer.contains("unknown") ||
                product == "sdk" ||
                product == "google_sdk" ||
                product.contains("emulator") ||
                product.contains("android_x86") ||
                hardware.contains("ranchu") ||
                hardware.contains("goldfish") ||
                hardware.contains("vbox") ||
                fingerprint.contains("generic")
    }

    private fun checkEmulatorFiles(): Boolean {
        return knownEmulatorFiles.any { File(it).exists() }
    }

    private fun checkPhoneNumber(): Boolean {
        return try {
            val process = Runtime.getRuntime().exec(arrayOf("getprop", "gsm.operator.alpha"))
            val reader = java.io.BufferedReader(
                java.io.InputStreamReader(process.inputStream)
            )
            val value = reader.readLine()
            process.destroy()
            value == "android"
        } catch (e: Exception) {
            false
        }
    }

    private fun checkEmulatorDrivers(): Boolean {
        return try {
            val drivers = File("/proc/tty/drivers")
            if (drivers.exists()) {
                val content = FileInputStream(drivers).bufferedReader().readText()
                content.contains("goldfish")
            } else false
        } catch (e: Exception) {
            false
        }
    }

    private fun checkQemuProps(): Boolean {
        return try {
            for (prop in knownEmulatorProperties) {
                val process = Runtime.getRuntime().exec(arrayOf("getprop", prop))
                val reader = java.io.BufferedReader(
                    java.io.InputStreamReader(process.inputStream)
                )
                val value = reader.readLine()
                process.destroy()
                if (value == "1") return true
            }
            false
        } catch (e: Exception) {
            false
        }
    }

    private fun checkGenymotion(): Boolean {
        return try {
            val process = Runtime.getRuntime().exec(arrayOf("getprop", "ro.product.manufacturer"))
            val reader = java.io.BufferedReader(
                java.io.InputStreamReader(process.inputStream)
            )
            val value = reader.readLine()?.lowercase()
            process.destroy()
            value == "genymotion"
        } catch (e: Exception) {
            false
        }
    }
}
