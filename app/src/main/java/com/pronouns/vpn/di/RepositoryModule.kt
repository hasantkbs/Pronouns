package com.pronouns.vpn.di

import android.content.Context
import com.pronouns.vpn.data.local.PreferencesStore
import com.pronouns.vpn.data.repository.AuthRepository
import com.pronouns.vpn.data.repository.AuthRepositoryImpl
import com.pronouns.vpn.data.repository.DeviceRepository
import com.pronouns.vpn.data.repository.DeviceRepositoryImpl
import com.pronouns.vpn.data.repository.UpdateRepository
import com.pronouns.vpn.data.repository.UpdateRepositoryImpl
import com.pronouns.vpn.data.repository.VpnRepository
import com.pronouns.vpn.data.repository.VpnRepositoryImpl
import dagger.Binds
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
abstract class RepositoryModule {

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
    abstract fun bindUpdateRepository(
        impl: UpdateRepositoryImpl
    ): UpdateRepository

    @Binds
    @Singleton
    abstract fun bindDeviceRepository(
        impl: DeviceRepositoryImpl
    ): DeviceRepository

    companion object {

        @Provides
        @Singleton
        fun providePreferencesStore(
            @ApplicationContext context: Context
        ): PreferencesStore = PreferencesStore(context)
    }
}
