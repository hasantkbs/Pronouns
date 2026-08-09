package com.pronouns.vpn.di

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.PreferenceDataStoreFactory
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.preferencesDataStoreFile
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import com.pronouns.vpn.data.local.EncryptedPrefsManager
import com.pronouns.vpn.data.local.KeystoreManager
import com.pronouns.vpn.data.local.VpnConfigManager
import com.pronouns.vpn.security.MemoryZeroizer
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideMasterKey(@ApplicationContext context: Context): MasterKey {
        return MasterKey.Builder(context)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
    }

    @Provides
    @Singleton
    fun provideEncryptedSharedPreferences(
        @ApplicationContext context: Context,
        masterKey: MasterKey
    ): androidx.security.crypto.EncryptedSharedPreferences {
        return androidx.security.crypto.EncryptedSharedPreferences.create(
            context,
            "vpn_secure_prefs",
            masterKey,
            androidx.security.crypto.EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            androidx.security.crypto.EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        ) as androidx.security.crypto.EncryptedSharedPreferences
    }

    @Provides
    @Singleton
    fun provideDataStore(@ApplicationContext context: Context): DataStore<Preferences> {
        return PreferenceDataStoreFactory.create {
            context.preferencesDataStoreFile("vpn_settings")
        }
    }

    @Provides
    @Singleton
    fun provideMemoryZeroizer(): MemoryZeroizer = MemoryZeroizer()

    @Provides
    @Singleton
    fun provideKeystoreManager(
        @ApplicationContext context: Context
    ): KeystoreManager = KeystoreManager(context)

    @Provides
    @Singleton
    fun provideEncryptedPrefsManager(
        encryptedSharedPreferences: EncryptedSharedPreferences
    ): EncryptedPrefsManager = EncryptedPrefsManager(encryptedSharedPreferences)

    @Provides
    @Singleton
    fun provideVpnConfigManager(
        @ApplicationContext context: Context,
        encryptedPrefsManager: EncryptedPrefsManager
    ): VpnConfigManager = VpnConfigManager(context, encryptedPrefsManager)
}
