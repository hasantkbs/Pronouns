package com.pronouns.vpn.di

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import com.pronouns.vpn.data.local.EncryptedPrefsManager
import com.pronouns.vpn.data.local.KeystoreManager
import com.pronouns.vpn.data.local.VpnConfigManager
import com.pronouns.vpn.data.repository.AuthRepositoryImpl
import com.pronouns.vpn.data.repository.DeviceRepositoryImpl
import com.pronouns.vpn.data.repository.UpdateRepositoryImpl
import com.pronouns.vpn.data.repository.VpnRepositoryImpl
import com.pronouns.vpn.domain.repository.AuthRepository
import com.pronouns.vpn.domain.repository.DeviceRepository
import com.pronouns.vpn.domain.repository.UpdateRepository
import com.pronouns.vpn.domain.repository.VpnRepository
import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
abstract class DatabaseModule {

    @Binds
    @Singleton
    abstract fun bindAuthRepository(
        impl: AuthRepositoryImpl
    ): AuthRepository

    @Binds
    @Singleton
    abstract fun bindVpnRepository(
        impl: VpnRepositoryImpl
    ): VpnRepository

    @Binds
    @Singleton
    abstract fun bindDeviceRepository(
        impl: DeviceRepositoryImpl
    ): DeviceRepository

    @Binds
    @Singleton
    abstract fun bindUpdateRepository(
        impl: UpdateRepositoryImpl
    ): UpdateRepository
}
