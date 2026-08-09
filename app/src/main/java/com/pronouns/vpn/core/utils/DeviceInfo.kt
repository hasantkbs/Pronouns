package com.pronouns.vpn.core.utils

import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.provider.Settings
import java.security.MessageDigest
import java.util.UUID

object DeviceInfo {

    fun generateDeviceId(context: Context): UUID {
        val prefs = context.getSharedPreferences("device_prefs", Context.MODE_PRIVATE)
        val stored = prefs.getString("device_id", null)
        if (stored != null) {
            return UUID.fromString(stored)
        }
        val id = UUID.randomUUID()
        prefs.edit().putString("device_id", id.toString()).apply()
        return id
    }

    fun getAndroidId(context: Context): String {
        return Settings.Secure.getString(
            context.contentResolver,
            Settings.Secure.ANDROID_ID
        ) ?: ""
    }

    fun getManufacturer(): String = Build.MANUFACTURER

    fun getModel(): String = Build.MODEL

    fun getAndroidVersion(): String = Build.VERSION.RELEASE

    fun getAppVersion(context: Context): String {
        return try {
            context.packageManager.getPackageInfo(context.packageName, 0).versionName ?: ""
        } catch (_: Exception) {
            ""
        }
    }

    fun getInstallTimestamp(context: Context): Long {
        return try {
            context.packageManager.getPackageInfo(context.packageName, 0).firstInstallTime
        } catch (_: Exception) {
            0L
        }
    }

    fun getAppSignatureHash(context: Context): String {
        return try {
            val info = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                context.packageManager.getPackageInfo(
                    context.packageName,
                    PackageManager.GET_SIGNING_CERTIFICATES
                )
            } else {
                @Suppress("DEPRECATION")
                context.packageManager.getPackageInfo(
                    context.packageName,
                    PackageManager.GET_SIGNATURES
                )
            }

            val digest = MessageDigest.getInstance("SHA-256")

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val signingInfo = info.signingInfo
                val certs = if (signingInfo.hasMultipleSigners()) {
                    signingInfo.getSigningCertificateHistory()
                } else {
                    signingInfo.getSigningCertificateHistory()
                }
                for (cert in certs) {
                    digest.update(cert.encoded)
                }
            } else {
                @Suppress("DEPRECATION")
                for (sig in info.signatures) {
                    digest.update(sig.toByteArray())
                }
            }

            digest.digest().joinToString("") { byte -> "%02x".format(byte) }
        } catch (_: Exception) {
            ""
        }
    }

    fun getDeviceAttestationPayload(context: Context): Map<String, String> {
        return mapOf(
            "device_id" to generateDeviceId(context).toString(),
            "android_id" to getAndroidId(context),
            "manufacturer" to getManufacturer(),
            "model" to getModel(),
            "android_version" to getAndroidVersion(),
            "app_version" to getAppVersion(context),
            "install_timestamp" to getInstallTimestamp(context).toString(),
            "signature_hash" to getAppSignatureHash(context)
        )
    }
}
