package com.pronouns.vpn.core.detection

import android.os.Build
import java.io.File
import java.io.RandomAccessFile
import java.net.Inet4Address
import java.net.NetworkInterface

object EmulatorDetector {

    private val knownEmulatorFingerprints = listOf(
        "google/sdk_gphone_arm64/generic",
        "google/sdk_gphone_arm/generic",
        "google/sdk_gphone_x86/generic",
        "google/sdk_gphone_x86_64/generic",
        "generic/sdk/generic",
        "generic_x86/sdk/generic",
        "generic_x86_64/sdk/generic"
    )

    private val knownEmulatorModels = listOf(
        "sdk_gphone64_arm64",
        "sdk_gphone64_x86_64",
        "sdk_gphone_arm",
        "sdk_gphone_x86",
        "sdk",
        "google_sdk",
        "emulator",
        "Android SDK built for x86",
        "Android SDK built for x86_64"
    )

    private val knownEmulatorManufacturers = listOf(
        "Google",
        "unknown"
    )

    private val knownEmulatorProducts = listOf(
        "sdk",
        "sdk_x86",
        "sdk_x86_64",
        "vbox86p",
        "generic",
        "generic_x86",
        "generic_x86_64"
    )

    private val knownEmulatorDevices = listOf(
        "generic",
        "generic_x86",
        "generic_x86_64",
        "emulator",
        "vbox86p"
    )

    private val knownEmulatorHardware = listOf(
        "ranchu",
        "goldfish",
        "vbox86"
    )

    private val emulatorProperties = listOf(
        "ro.kernel.qemu",
        "ro.hardware.ranchu",
        "ro.hardware.goldfish",
        "ro.emulator",
        "ro.leap.td"
    )

    private val emulatorFiles = listOf(
        "/system/lib/emudrv",
        "/system/lib64/emudrv",
        "/dev/socket/qemud",
        "/dev/qemu_pipe"
    )

    private val defaultEmulatorNumber = "15555215554"

    private const val EMULATOR_IP_PREFIX = "10.0.2."

    fun isEmulator(): Boolean {
        if (checkFingerprint()) return true
        if (checkModel()) return true
        if (checkManufacturer()) return true
        if (checkProduct()) return true
        if (checkDevice()) return true
        if (checkHardware()) return true
        if (checkQemuDriver()) return true
        if (checkProperties()) return true
        if (checkEmulatorFiles()) return true
        if (checkPhoneNumber()) return true
        if (checkEmulatorIp()) return true
        return false
    }

    private fun checkFingerprint(): Boolean {
        val fingerprint = Build.FINGERPRINT ?: return false
        return knownEmulatorFingerprints.any { fingerprint.startsWith(it) }
    }

    private fun checkModel(): Boolean {
        val model = Build.MODEL ?: return false
        return knownEmulatorModels.any { model.startsWith(it) }
    }

    private fun checkManufacturer(): Boolean {
        val manufacturer = Build.MANUFACTURER ?: return false
        return knownEmulatorManufacturers.contains(manufacturer)
    }

    private fun checkProduct(): Boolean {
        val product = Build.PRODUCT ?: return false
        return knownEmulatorProducts.contains(product)
    }

    private fun checkDevice(): Boolean {
        val device = Build.DEVICE ?: return false
        return knownEmulatorDevices.contains(device)
    }

    private fun checkHardware(): Boolean {
        val hardware = Build.HARDWARE ?: return false
        return knownEmulatorHardware.any { hardware.contains(it) }
    }

    private fun checkQemuDriver(): Boolean {
        return try {
            val file = File("/dev/tts/0")
            if (file.exists()) {
                val raf = RandomAccessFile(file, "r")
                val content = ByteArray(1024)
                raf.read(content)
                raf.close()
                val str = String(content)
                str.contains("QEMU")
            } else {
                false
            }
        } catch (_: Exception) {
            false
        }
    }

    private fun checkProperties(): Boolean {
        return emulatorProperties.any { prop ->
            try {
                val process = Runtime.getRuntime().exec("getprop $prop")
                val reader = java.io.BufferedReader(
                    java.io.InputStreamReader(process.inputStream)
                )
                val value = reader.readLine()
                reader.close()
                process.waitFor()
                value != null && value.isNotEmpty() && value != "0" && value != "false"
            } catch (_: Exception) {
                false
            }
        }
    }

    private fun checkEmulatorFiles(): Boolean {
        return emulatorFiles.any { path -> File(path).exists() }
    }

    private fun checkPhoneNumber(): Boolean {
        return try {
            val contextClass = Class.forName("android.app.ActivityThread")
            val method = contextClass.getMethod("currentApplication")
            val context = method.invoke(null) as? android.content.Context ?: return false
            val tm = context.getSystemService(android.content.Context.TELEPHONY_SERVICE)
                as? android.telephony.TelephonyManager ?: return false
            val line1Number = tm.line1Number
            line1Number == defaultEmulatorNumber
        } catch (_: Exception) {
            false
        }
    }

    private fun checkEmulatorIp(): Boolean {
        return try {
            val interfaces = NetworkInterface.getNetworkInterfaces()
            while (interfaces.hasMoreElements()) {
                val networkInterface = interfaces.nextElement()
                val addresses = networkInterface.inetAddresses
                while (addresses.hasMoreElements()) {
                    val address = addresses.nextElement()
                    if (address is Inet4Address && address.isLoopbackAddress) continue
                    if (address is Inet4Address && address.hostAddress?.startsWith(EMULATOR_IP_PREFIX) == true) {
                        return true
                    }
                }
            }
            false
        } catch (_: Exception) {
            false
        }
    }
}
