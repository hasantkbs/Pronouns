package com.pronouns.vpn.core.detection

import android.os.Build
import java.io.File

object RootDetector {

    private val rootBinaryPaths = listOf(
        "/sbin/su",
        "/system/bin/su",
        "/system/xbin/su",
        "/system/sbin/su",
        "/magisk/.core/bin/su"
    )

    private val rootApps = listOf(
        "com.topjohnwu.magisk",
        "com.thirdway.smartcharge",
        "eu.chainfire.supersu",
        "com.noshufou.android.su",
        "com.koushikdutta.superuser",
        "com.zachspong.temprootremovejb",
        "com.ramdroid.appquarantine"
    )

    private val rootPaths = listOf(
        "/system/app/Superuser.apk",
        "/system/app/SuperSU.apk",
        "/system/app/Magisk.apk",
        "/data/data/eu.chainfire.supersu"
    )

    fun isRooted(): Boolean {
        if (checkRootBinaryPaths()) return true
        if (checkBuildTags()) return true
        if (checkRootApps()) return true
        if (checkRootPaths()) return true
        return false
    }

    private fun checkRootBinaryPaths(): Boolean {
        return rootBinaryPaths.any { path -> File(path).exists() }
    }

    private fun checkBuildTags(): Boolean {
        val tags = Build.TAGS ?: return false
        return tags.contains("test-keys")
    }

    private fun checkRootApps(): Boolean {
        return rootApps.any { packageName ->
            try {
                Class.forName("android.app.ActivityManager")
                val pm = try {
                    val contextClass = Class.forName("android.app.ActivityThread")
                    val method = contextClass.getMethod("currentApplication")
                    val context = method.invoke(null) as? android.content.Context ?: return@any false
                    context.packageManager
                } catch (_: Exception) {
                    return@any false
                }
                try {
                    pm.getPackageInfo(packageName, 0)
                    true
                } catch (_: Exception) {
                    false
                }
            } catch (_: Exception) {
                false
            }
        }
    }

    private fun checkRootPaths(): Boolean {
        return rootPaths.any { path -> File(path).exists() }
    }
}
