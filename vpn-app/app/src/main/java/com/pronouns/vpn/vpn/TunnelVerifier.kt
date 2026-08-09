package com.pronouns.vpn.vpn

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import java.io.IOException
import java.net.HttpURLConnection
import java.net.InetSocketAddress
import java.net.Socket
import java.net.URL
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class TunnelVerifier @Inject constructor(
    private val context: Context
) {
    private val verifyEndpoints = listOf(
        "https://api.pronouns.app/v1/health",
        "https://8.8.8.8"
    )
    private val blockedEndpoints = listOf(
        "http://ifconfig.me",
        "http://ip-api.com"
    )

    suspend fun verifyTunnel(): Boolean {
        return verifyDnsLeak() && verifyBackendReachable() && verifyBlockedEndpoints()
    }

    private suspend fun verifyDnsLeak(): Boolean {
        return try {
            val socket = Socket()
            socket.connect(InetSocketAddress("1.1.1.1", 53), 5000)
            socket.close()
            true
        } catch (e: Exception) {
            false
        }
    }

    private suspend fun verifyBackendReachable(): Boolean {
        for (endpoint in verifyEndpoints) {
            try {
                val url = URL(endpoint)
                val connection = url.openConnection() as HttpURLConnection
                connection.connectTimeout = 5000
                connection.readTimeout = 5000
                connection.instanceFollowRedirects = false
                val responseCode = connection.responseCode
                connection.disconnect()
                if (responseCode in 200..499) return true
            } catch (e: IOException) {
                if (endpoint == verifyEndpoints.last()) return false
            }
        }
        return false
    }

    private suspend fun verifyBlockedEndpoints(): Boolean {
        for (endpoint in blockedEndpoints) {
            try {
                val url = URL(endpoint)
                val connection = url.openConnection() as HttpURLConnection
                connection.connectTimeout = 3000
                connection.readTimeout = 3000
                val responseCode = connection.responseCode
                connection.disconnect()
                if (responseCode in 200..299) return false
            } catch (e: IOException) {
                continue
            }
        }
        return true
    }

    fun isVpnActive(): Boolean {
        val connectivityManager = context.getSystemService(Context.CONNECTIVITY_SERVICE)
                as ConnectivityManager
        val activeNetwork = connectivityManager.activeNetwork ?: return false
        val caps = connectivityManager.getNetworkCapabilities(activeNetwork) ?: return false
        return caps.hasTransport(NetworkCapabilities.TRANSPORT_VPN)
    }

    fun registerVpnMonitor(callback: (Boolean) -> Unit) {
        val connectivityManager = context.getSystemService(Context.CONNECTIVITY_SERVICE)
                as ConnectivityManager
        val networkCallback = object : ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: Network) {
                val caps = connectivityManager.getNetworkCapabilities(network)
                callback(caps?.hasTransport(NetworkCapabilities.TRANSPORT_VPN) == true)
            }

            override fun onLost(network: Network) {
                callback(false)
            }

            override fun onCapabilitiesChanged(
                network: Network,
                networkCapabilities: NetworkCapabilities
            ) {
                callback(networkCapabilities.hasTransport(NetworkCapabilities.TRANSPORT_VPN))
            }
        }

        val request = NetworkRequest.Builder()
            .addTransportType(NetworkCapabilities.TRANSPORT_VPN)
            .build()
        connectivityManager.registerNetworkCallback(request, networkCallback)
    }
}
