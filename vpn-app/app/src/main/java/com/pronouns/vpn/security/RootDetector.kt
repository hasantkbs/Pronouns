package com.pronouns.vpn.security

import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class RootDetector @Inject constructor() {

    private val knownRootPaths = listOf(
        "/system/app/Superuser.apk",
        "/system/app/su.apk",
        "/sbin/su",
        "/system/bin/su",
        "/system/xbin/su",
        "/data/local/xbin/su",
        "/data/local/bin/su",
        "/system/sd/xbin/su",
        "/system/bin/failsafe/su",
        "/data/local/su",
        "/su/bin/su"
    )

    private val knownRootPackages = listOf(
        "com.noshufou.android.su",
        "com.noshufou.android.su.elite",
        "eu.chainfire.supersu",
        "com.koushikdutta.superuser",
        "com.thirdparty.superuser",
        "com.yellowes.su",
        "com.topjohnwu.magisk",
        "io.github.vvb2060.magisk",
        "com.kingroot.kinguser",
        "com.kingo.root",
        "com.superuser.kinguser"
    )

    private val suspiciousCommands = listOf(
        "su", "busybox", "magisk", "supolicy"
    )

    fun isDeviceRooted(): Boolean {
        return checkBuildTags() ||
                checkTestKeys() ||
                checkRootPaths() ||
                checkRootPackages() ||
                checkSuAccess()
    }

    private fun checkBuildTags(): Boolean {
        val tags = android.os.Build.TAGS
        return tags != null && tags.contains("test-keys")
    }

    private fun checkTestKeys(): Boolean {
        return android.os.Build.FINGERPRINT.contains("test-keys")
    }

    private fun checkRootPaths(): Boolean {
        return knownRootPaths.any { File(it).exists() }
    }

    private fun checkRootPackages(): Boolean {
        return try {
            val pm = com.pronouns.vpn.VpnApplication.instance?.packageManager
            if (pm == null) {
                knownRootPaths.any { File(it).exists() }
            } else {
                knownRootPackages.any { pkg ->
                    try {
                        pm.getPackageInfo(pkg, 0)
                        true
                    } catch (e: android.content.pm.PackageManager.NameNotFoundException) {
                        false
                    }
                }
            }
        } catch (e: Exception) {
            false
        }
    }

    private fun checkSuAccess(): Boolean {
        return try {
            val process = Runtime.getRuntime().exec("which su")
            val reader = java.io.BufferedReader(
                java.io.InputStreamReader(process.inputStream)
            )
            val line = reader.readLine()
            process.destroy()
            line != null && line.isNotEmpty()
        } catch (e: Exception) {
            false
        }
    }
}
