package com.pronouns.vpn.vpn

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Intent
import android.net.VpnService
import android.os.Build
import android.os.ParcelFileDescriptor
import androidx.core.app.NotificationCompat
import com.pronouns.vpn.MainActivity
import com.pronouns.vpn.R
import com.pronouns.vpn.data.local.VpnConfigManager
import com.pronouns.vpn.domain.model.VpnStatus
import dagger.hilt.android.AndroidEntryPoint
import java.io.FileInputStream
import java.io.FileOutputStream
import javax.inject.Inject

@AndroidEntryPoint
class VpnConnectionService : VpnService() {

    @Inject
    lateinit var vpnConfigManager: VpnConfigManager

    @Inject
    lateinit var tunnelVerifier: TunnelVerifier

    private var vpnInterface: ParcelFileDescriptor? = null
    private var isRunning = false

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, createNotification("Initializing VPN..."))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent == null) {
            stopSelf()
            return START_NOT_STICKY
        }

        val username = intent.getStringExtra(EXTRA_USERNAME) ?: run {
            stopSelf()
            return START_NOT_STICKY
        }
        val password = intent.getStringExtra(EXTRA_PASSWORD) ?: run {
            stopSelf()
            return START_NOT_STICKY
        }

        establishVpnConnection(username, password)
        return START_STICKY
    }

    override fun onDestroy() {
        isRunning = false
        vpnInterface?.close()
        super.onDestroy()
    }

    private fun establishVpnConnection(username: String, password: String) {
        try {
            val builder = Builder()

            builder.setName("PronounsVPN")
            builder.setMtu(1500)

            builder.addAddress("10.8.0.2", 24)
            builder.addRoute("0.0.0.0", 0)
            builder.addDnsServer("1.1.1.1")
            builder.addDnsServer("8.8.8.8")

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                builder.setMetered(false)
            }

            builder.setBlocking(true)

            val encryptedConfig = vpnConfigManager.loadEncryptedConfig()
            if (encryptedConfig != null) {
                parseAndConfigure(builder, encryptedConfig)
            }

            vpnInterface = builder.establish()
            isRunning = vpnInterface != null

            if (isRunning) {
                startForeground(NOTIFICATION_ID, createNotification("VPN Connected"))
                tunnelVerifier.verifyTunnel()
                startPacketCapture()
            }
        } catch (e: Exception) {
            android.util.Log.e("VpnService", "VPN establishment failed", e)
            stopSelf()
        }
    }

    private fun parseAndConfigure(builder: Builder, configData: ByteArray) {
        val configStr = String(configData, Charsets.UTF_8)
        configStr.lines().forEach { line ->
            val trimmed = line.trim()
            when {
                trimmed.startsWith("remote ") -> {
                    val parts = trimmed.split("\\s+".toRegex())
                    if (parts.size >= 3) {
                        try {
                            val port = parts[2].toInt()
                            builder.addServer(parts[1], port)
                        } catch (e: NumberFormatException) {
                            builder.addServer(parts[1], 443)
                        }
                    }
                }
                trimmed.startsWith("proto ") -> {
                    // Protocol parsed by OpenVPN library
                }
            }
        }
    }

    private fun startPacketCapture() {
        Thread {
            try {
                val input = FileInputStream(vpnInterface?.fileDescriptor)
                val output = FileOutputStream(vpnInterface?.fileDescriptor)
                val packet = ByteArray(32767)

                while (isRunning) {
                    val length = input.read(packet)
                    if (length > 0) {
                        output.write(packet, 0, length)
                    }
                }
            } catch (e: Exception) {
                android.util.Log.d("VpnService", "Packet capture ended")
            }
        }.start()
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "VPN Connection",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Shows VPN connection status"
            setShowBadge(false)
        }
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(channel)
    }

    private fun createNotification(text: String): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("PronounsVPN")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_lock_lock)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    companion object {
        const val EXTRA_USERNAME = "vpn_username"
        const val EXTRA_PASSWORD = "vpn_password"
        private const val CHANNEL_ID = "vpn_connection_channel"
        private const val NOTIFICATION_ID = 1001
    }
}
