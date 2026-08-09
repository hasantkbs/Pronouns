package com.pronouns.vpn.core.security

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import com.pronouns.vpn.core.utils.Zeroizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class SecureCredentialManager(
    private val context: Context,
    private val androidKeystoreWrapper: AndroidKeystoreWrapper,
    private val cryptoManager: CryptoManager
) {
    private val prefs: SharedPreferences by lazy {
        val masterKey = MasterKey.Builder(context)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
        EncryptedSharedPreferences.create(
            context,
            PREFS_NAME,
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        )
    }

    suspend fun storeVpnCredentials(alias: String, username: String, password: String) {
        withContext(Dispatchers.IO) {
            if (!androidKeystoreWrapper.containsKey(alias)) {
                androidKeystoreWrapper.generateKey(alias)
            }
            val credentials = "$username:$password"
            val encrypted = cryptoManager.encryptString(alias, credentials)
            prefs.edit()
                .putString(PREF_ACTIVE_VPN_ALIAS, alias)
                .putString(PREF_VPN_CIPHERTEXT, encrypted)
                .apply()
            Zeroizer.zeroizeString(credentials)
        }
    }

    suspend fun getVpnCredentials(alias: String): Pair<String, String>? {
        return withContext(Dispatchers.IO) {
            val storedAlias = prefs.getString(PREF_ACTIVE_VPN_ALIAS, null)
            if (storedAlias != alias) return@withContext null
            val ciphertext = prefs.getString(PREF_VPN_CIPHERTEXT, null) ?: return@withContext null
            try {
                val decrypted = cryptoManager.decryptString(alias, ciphertext)
                val parts = decrypted.split(":", limit = 2)
                if (parts.size != 2) {
                    Zeroizer.zeroizeString(decrypted)
                    return@withContext null
                }
                val result = Pair(parts[0], parts[1])
                Zeroizer.zeroizeString(decrypted)
                result
            } catch (_: Exception) {
                null
            }
        }
    }

    suspend fun storeAuthToken(token: String) {
        withContext(Dispatchers.IO) {
            if (!androidKeystoreWrapper.containsKey(AUTH_TOKEN_KEY_ALIAS)) {
                androidKeystoreWrapper.generateKey(AUTH_TOKEN_KEY_ALIAS)
            }
            val encrypted = cryptoManager.encryptString(AUTH_TOKEN_KEY_ALIAS, token)
            prefs.edit()
                .putString(PREF_AUTH_CIPHERTEXT, encrypted)
                .apply()
        }
    }

    suspend fun getAuthToken(): String? {
        return withContext(Dispatchers.IO) {
            val ciphertext = prefs.getString(PREF_AUTH_CIPHERTEXT, null) ?: return@withContext null
            try {
                cryptoManager.decryptString(AUTH_TOKEN_KEY_ALIAS, ciphertext)
            } catch (_: Exception) {
                null
            }
        }
    }

    suspend fun clearAll() {
        withContext(Dispatchers.IO) {
            val vpnAlias = prefs.getString(PREF_ACTIVE_VPN_ALIAS, null)
            if (vpnAlias != null) {
                androidKeystoreWrapper.deleteKey(vpnAlias)
            }
            androidKeystoreWrapper.deleteKey(AUTH_TOKEN_KEY_ALIAS)
            prefs.edit().clear().apply()
        }
    }

    suspend fun rotateVpnCredentials(newAlias: String) {
        withContext(Dispatchers.IO) {
            val oldAlias = prefs.getString(PREF_ACTIVE_VPN_ALIAS, null) ?: return@withContext
            val ciphertext = prefs.getString(PREF_VPN_CIPHERTEXT, null) ?: return@withContext
            try {
                val decrypted = cryptoManager.decryptString(oldAlias, ciphertext)
                androidKeystoreWrapper.deleteKey(oldAlias)
                androidKeystoreWrapper.generateKey(newAlias)
                val newEncrypted = cryptoManager.encryptString(newAlias, decrypted)
                prefs.edit()
                    .putString(PREF_ACTIVE_VPN_ALIAS, newAlias)
                    .putString(PREF_VPN_CIPHERTEXT, newEncrypted)
                    .apply()
                Zeroizer.zeroizeString(decrypted)
            } catch (_: Exception) {
            }
        }
    }

    suspend fun hasCredentials(): Boolean {
        return withContext(Dispatchers.IO) {
            val alias = prefs.getString(PREF_ACTIVE_VPN_ALIAS, null)
            alias != null && prefs.contains(PREF_VPN_CIPHERTEXT)
        }
    }

    suspend fun hasAuthToken(): Boolean {
        return withContext(Dispatchers.IO) {
            prefs.contains(PREF_AUTH_CIPHERTEXT)
        }
    }

    private companion object {
        private const val PREFS_NAME = "secure_cred_prefs"
        private const val PREF_ACTIVE_VPN_ALIAS = "active_vpn_alias"
        private const val PREF_VPN_CIPHERTEXT = "vpn_ciphertext"
        private const val PREF_AUTH_CIPHERTEXT = "auth_ciphertext"
        private const val AUTH_TOKEN_KEY_ALIAS = "auth_token_key"
    }
}
