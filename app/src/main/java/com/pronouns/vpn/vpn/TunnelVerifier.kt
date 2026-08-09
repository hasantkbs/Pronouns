package com.pronouns.vpn.vpn

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.VpnService
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.net.HttpURLConnection
import java.net.InetAddress
import java.net.NetworkInterface
import java.net.URL
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class TunnelVerifier @Inject constructor(
    @ApplicationContext private val context: Context
) {

    fun isTunnelEstablished(): Boolean {
        return getTunnelInterface() != null
    }

    suspend fun verifyTunnel(
        maxRetries: Int = 10,
        delayMs: Long = 1000
    ): Boolean = withContext(Dispatchers.IO) {
        repeat(maxRetries) {
            if (isTunnelEstablished()) return@withContext true
            delay(delayMs)
        }
        isTunnelEstablished()
    }

    suspend fun verifyBackendReachable(url: String): Boolean = withContext(Dispatchers.IO) {
        val vpnNetwork = resolveVpnNetwork() ?: return@withContext false

        try {
            val connection = vpnNetwork.openConnection(URL(url)) as HttpURLConnection
            connection.connectTimeout = 5000
            connection.readTimeout = 5000
            connection.requestMethod = "HEAD"
            connection.instanceFollowRedirects = false

            val code = try {
                connection.connect()
                connection.responseCode
            } finally {
                connection.disconnect()
            }

            code in 200..499
        } catch (_: Exception) {
            false
        }
    }

    suspend fun verifyDnsResolution(hostname: String): Boolean = withContext(Dispatchers.IO) {
        try {
            val address = InetAddress.getByName(hostname)
            !address.isLoopbackAddress && !address.isAnyLocalAddress
        } catch (_: Exception) {
            false
        }
    }

    fun getTunnelInterface(): String? {
        return try {
            val interfaces = NetworkInterface.getNetworkInterfaces()
            while (interfaces.hasMoreElements()) {
                val iface = interfaces.nextElement()
                if ((iface.name.startsWith("tun") || iface.name.startsWith("tap")) &&
                    iface.isUp && !iface.isLoopback
                ) {
                    return iface.name
                }
            }
            null
        } catch (_: Exception) {
            null
        }
    }

    fun verify() {
        if (!isTunnelEstablished()) {
            throw TunnelNotEstablishedException("VPN tunnel is not established")
        }
    }

    private fun resolveVpnNetwork(): Network? {
        val connectivityManager =
            context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager

        for (network in connectivityManager.allNetworks) {
            val caps = connectivityManager.getNetworkCapabilities(network) ?: continue
            if (caps.hasTransport(NetworkCapabilities.TRANSPORT_VPN)) {
                return network
            }
        }
        return null
    }

    class TunnelNotEstablishedException(message: String) : Exception(message)
}
