package com.pronouns.vpn.domain.usecase

import com.pronouns.vpn.core.security.SecureCredentialManager
import com.pronouns.vpn.data.local.PreferencesStore
import com.pronouns.vpn.data.repository.VpnRepository
import com.pronouns.vpn.domain.model.VpnCredentials
import kotlinx.coroutines.flow.first
import javax.inject.Inject

class ConnectVpnUseCase @Inject constructor(
    private val vpnRepository: VpnRepository,
    private val secureCredentialManager: SecureCredentialManager,
    private val preferencesStore: PreferencesStore
) {
    suspend operator fun invoke(): Result<Unit> {
        return runCatching {
            val profile = vpnRepository.getVpnProfile().getOrThrow()
            val creds = resolveCredentials()

            vpnRepository.connectVpn(profile, creds).getOrThrow()
        }
    }

    suspend fun disconnect(): Result<Unit> {
        return vpnRepository.disconnectVpn()
    }

    private suspend fun resolveCredentials(): VpnCredentials {
        val stored = secureCredentialManager.getVpnCredentials(VPN_CREDS_ALIAS)

        if (stored != null) {
            val lastRotation = preferencesStore.getLastVpnRotation().first() ?: 0L
            val age = System.currentTimeMillis() - lastRotation
            if (age < TTL_MILLIS) {
                return VpnCredentials(stored.first, stored.second)
            }
        }

        val rotated = vpnRepository.rotateCredentials().getOrThrow()
        secureCredentialManager.storeVpnCredentials(
            VPN_CREDS_ALIAS,
            rotated.username,
            rotated.password
        )
        return rotated
    }

    private companion object {
        private const val VPN_CREDS_ALIAS = "vpn_active_creds"
        private const val TTL_HOURS = 6L
        private const val TTL_MILLIS = TTL_HOURS * 60 * 60 * 1000L
    }
}
