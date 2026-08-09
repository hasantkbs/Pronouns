package com.pronouns.vpn.update

import android.content.Context
import android.content.Intent
import android.content.pm.PackageInstaller
import android.content.pm.PackageManager
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class UpdateInstaller @Inject constructor(
    private val context: Context
) {
    private val packageInstaller = context.packageManager.packageInstaller

    fun installApk(apkFile: File): Boolean {
        return try {
            val uri = FileProvider.getUriForFile(
                context,
                "${context.packageName}.fileprovider",
                apkFile
            )

            val intent = Intent(Intent.ACTION_VIEW).apply {
                setDataAndType(uri, "application/vnd.android.package-archive")
                flags = Intent.FLAG_GRANT_READ_URI_PERMISSION or
                        Intent.FLAG_ACTIVITY_NEW_TASK
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }

            context.startActivity(intent)
            true
        } catch (e: Exception) {
            android.util.Log.e("UpdateInstaller", "Install failed", e)
            false
        }
    }

    fun installSession(apkFile: File, onComplete: ((Boolean) -> Unit)? = null): Boolean {
        return try {
            val sessionParams = PackageInstaller.SessionParams(
                PackageInstaller.SessionParams.MODE_FULL_INSTALL
            ).apply {
                setAppPackageName(context.packageName)
                setSize(apkFile.length())
            }

            val sessionId = packageInstaller.createSession(sessionParams)
            val session = packageInstaller.openSession(sessionId)

            FileInputStream(apkFile).use { input ->
                val output = session.openWrite("base.apk", 0, apkFile.length())
                input.copyTo(output)
                session.fsync(output)
                session.close()
            }

            val pendingIntent = android.app.PendingIntent.getBroadcast(
                context,
                sessionId,
                Intent(context, InstallReceiver::class.java),
                android.app.PendingIntent.FLAG_UPDATE_CURRENT or
                        android.app.PendingIntent.FLAG_IMMUTABLE
            )

            val statusReceiver = if (onComplete != null) {
                android.app.PendingIntent.getBroadcast(
                    context,
                    sessionId,
                    Intent(context, InstallReceiver::class.java).apply {
                        putExtra("callback", object : android.os.ResultReceiver(null) {
                            override fun onReceiveResult(resultCode: Int, resultData: android.os.Bundle?) {
                                onComplete(resultCode == PackageInstaller.STATUS_SUCCESS)
                            }
                        })
                    },
                    android.app.PendingIntent.FLAG_UPDATE_CURRENT or
                            android.app.PendingIntent.FLAG_IMMUTABLE
                )
            } else pendingIntent

            session.commit(statusReceiver.intentSender)
            true
        } catch (e: Exception) {
            android.util.Log.e("UpdateInstaller", "Session install failed", e)
            false
        }
    }

    class InstallReceiver : android.content.BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            val status = intent.getIntExtra(
                PackageInstaller.EXTRA_STATUS,
                PackageInstaller.STATUS_FAILURE
            )

            when (status) {
                PackageInstaller.STATUS_SUCCESS -> {
                    android.util.Log.i("UpdateInstaller", "Install succeeded")
                }
                PackageInstaller.STATUS_FAILURE -> {
                    val message = intent.getStringExtra(
                        PackageInstaller.EXTRA_STATUS_MESSAGE
                    ) ?: "Unknown error"
                    android.util.Log.e("UpdateInstaller", "Install failed: $message")
                }
            }
        }
    }
}
