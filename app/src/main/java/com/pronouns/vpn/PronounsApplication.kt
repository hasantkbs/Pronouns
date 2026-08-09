package com.pronouns.vpn

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.os.Build
import androidx.hilt.work.HiltWorkerFactory
import androidx.work.Configuration
import dagger.hilt.android.HiltAndroidApp
import javax.inject.Inject

@HiltAndroidApp
class PronounsApplication : Application(), Configuration.Provider {

    @Inject
    lateinit var workerFactory: HiltWorkerFactory

    override fun onCreate() {
        super.onCreate()
        initializeSecurityProviders()
        createNotificationChannels()
    }

    override val workManagerConfiguration: Configuration
        get() = Configuration.Builder()
            .setWorkerFactory(workerFactory)
            .build()

    private fun initializeSecurityProviders() {
        try {
            val clazz = Class.forName("com.google.android.gms.security.ProviderInstaller")
            clazz.getMethod("installIfNeeded", android.content.Context::class.java)
                .invoke(null, applicationContext)
        } catch (_: Exception) {
        }
    }

    private fun createNotificationChannels() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val manager = getSystemService(NotificationManager::class.java)

            val vpnChannel = NotificationChannel(
                VPN_CHANNEL_ID,
                "VPN Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "VPN connection status notifications"
                setShowBadge(false)
            }

            val updateChannel = NotificationChannel(
                UPDATE_CHANNEL_ID,
                "App Updates",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Application update notifications"
            }

            manager.createNotificationChannel(vpnChannel)
            manager.createNotificationChannel(updateChannel)
        }
    }

    companion object {
        const val VPN_CHANNEL_ID = "pronouns_vpn_channel"
        const val UPDATE_CHANNEL_ID = "pronouns_update_channel"
    }
}
