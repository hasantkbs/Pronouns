package com.pronouns.vpn.domain.usecase

import com.pronouns.vpn.data.remote.ApiService
import com.pronouns.vpn.vpn.TunnelVerifier
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.net.InetAddress
import java.net.URL
import javax.inject.Inject

class VerifyTunnelUseCase @Inject constructor(
    private val tunnelVerifier: TunnelVerifier,
    private val apiService: ApiService
) {
    suspend operator fun invoke(timeoutSeconds: Long = 30): Result<Boolean> {
        return runCatching {
            waitForTunnel(timeoutSeconds)
            verifyBackendReachability()
            verifyDnsResolution()
            true
        }
    }

    private suspend fun waitForTunnel(timeoutSeconds: Long) {
        val deadline = System.currentTimeMillis() + (timeoutSeconds * 1000)
        var lastError: Exception? = null

        while (System.currentTimeMillis() < deadline) {
            try {
                tunnelVerifier.verify()
                return
            } catch (e: Exception) {
                lastError = e
                delay(POLL_INTERVAL_MS)
            }
        }

        throw lastError ?: TunnelTimeoutException("Tunnel not established within ${timeoutSeconds}s")
    }

    private suspend fun verifyBackendReachability() {
        withContext(Dispatchers.IO) {
            val response = apiService.healthCheck()
            if (!response.isSuccessful) {
                throw BackendUnreachableException(
                    "Health check failed: HTTP ${response.code()}"
                )
            }
        }
    }

    private suspend fun verifyDnsResolution() {
        withContext(Dispatchers.IO) {
            try {
                val address = InetAddress.getByName(DNS_CHECK_HOST)
                if (address.isLoopbackAddress || address.isAnyLocalAddress) {
                    throw DnsResolutionException("DNS resolution returned local address")
                }
            } catch (e: Exception) {
                throw DnsResolutionException("DNS resolution failed: ${e.message}", e)
            }
        }
    }

    private class TunnelTimeoutException(message: String) : Exception(message)
    private class BackendUnreachableException(message: String) : Exception(message)
    private class DnsResolutionException(message: String, cause: Throwable? = null) : Exception(message, cause)

    private companion object {
        private const val POLL_INTERVAL_MS = 500L
        private const val DNS_CHECK_HOST = "google.com"
    }
}
