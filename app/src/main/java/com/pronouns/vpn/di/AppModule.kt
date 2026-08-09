package com.pronouns.vpn.di

import android.content.Context
import com.pronouns.vpn.core.security.AndroidKeystoreWrapper
import com.pronouns.vpn.core.security.CertificatePinnerBuilder
import com.pronouns.vpn.core.security.CryptoManager
import com.pronouns.vpn.core.security.SecureCredentialManager
import com.pronouns.vpn.core.security.StringEncryption
import com.pronouns.vpn.core.utils.DeviceInfo
import com.pronouns.vpn.core.utils.Zeroizer
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Qualifier
import javax.inject.Singleton

@Qualifier
@Retention(AnnotationRetention.BINARY)
annotation class AppContext

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideZeroizer(): Zeroizer = Zeroizer

    @Provides
    @Singleton
    fun provideStringEncryption(): StringEncryption = StringEncryption

    @Provides
    @Singleton
    fun provideAndroidKeystoreWrapper(
        @ApplicationContext context: Context
    ): AndroidKeystoreWrapper = AndroidKeystoreWrapper(context)

    @Provides
    @Singleton
    fun provideCryptoManager(
        androidKeystoreWrapper: AndroidKeystoreWrapper
    ): CryptoManager = CryptoManager(androidKeystoreWrapper)

    @Provides
    @Singleton
    fun provideSecureCredentialManager(
        @ApplicationContext context: Context,
        androidKeystoreWrapper: AndroidKeystoreWrapper,
        cryptoManager: CryptoManager
    ): SecureCredentialManager = SecureCredentialManager(
        context,
        androidKeystoreWrapper,
        cryptoManager
    )

    @Provides
    @Singleton
    fun provideCertificatePinner(
        @ApplicationContext context: Context
    ): okhttp3.CertificatePinner =
        CertificatePinnerBuilder.buildCertificatePinner(context)

    @Provides
    @Singleton
    fun provideDeviceInfo(): DeviceInfo = DeviceInfo

    @Provides
    @AppContext
    fun provideAppContext(
        @ApplicationContext context: Context
    ): Context = context
}
