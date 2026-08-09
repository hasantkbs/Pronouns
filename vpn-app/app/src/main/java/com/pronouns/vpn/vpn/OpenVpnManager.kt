package com.pronouns.vpn.vpn

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import net.openvpn.openvpn.OpenVpnService
import net.openvpn.openvpn.OpenVpnStatusListener
import java.util.concurrent.atomic.AtomicBoolean
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class OpenVpnManager @Inject constructor(
    private val context: Context
) {
    private var vpnService: OpenVpnService? = null
    private val isBound = AtomicBoolean(false)
    private var statusListener: OpenVpnStatusListener? = null

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName, service: IBinder) {
            vpnService = (service as OpenVpnService.LocalBinder).getService()
            isBound.set(true)
        }

        override fun onServiceDisconnected(name: ComponentName) {
            vpnService = null
            isBound.set(false)
        }
    }

    fun startVpn(username: String, password: String) {
        val intent = Intent(context, VpnConnectionService::class.java).apply {
            putExtra(VpnConnectionService.EXTRA_USERNAME, username)
            putExtra(VpnConnectionService.EXTRA_PASSWORD, password)
        }
        context.startForegroundService(intent)
    }

    fun stopVpn() {
        val intent = Intent(context, VpnConnectionService::class.java)
        context.stopService(intent)
    }

    fun setStatusListener(listener: OpenVpnStatusListener?) {
        statusListener = listener
    }

    fun bindService() {
        val intent = Intent(context, OpenVpnService::class.java)
        context.bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    fun unbindService() {
        if (isBound.get()) {
            context.unbindService(serviceConnection)
            isBound.set(false)
        }
    }

    fun isRunning(): Boolean {
        return vpnService?.isRunning ?: false
    }

    fun destroy() {
        unbindService()
        vpnService = null
    }
}
