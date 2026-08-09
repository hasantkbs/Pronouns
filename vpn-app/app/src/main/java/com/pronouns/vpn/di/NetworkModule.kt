package com.pronouns.vpn.di

import com.pronouns.vpn.data.remote.api.BootstrapApi
import com.pronouns.vpn.data.remote.api.DeviceApi
import com.pronouns.vpn.data.remote.api.UpdateApi
import com.pronouns.vpn.data.remote.interceptor.AuthInterceptor
import com.pronouns.vpn.security.CertificatePinner
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import okhttp3.CertificatePinner as OkHttpCertificatePinner
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.moshi.MoshiConverterFactory
import java.security.KeyStore
import java.security.cert.CertificateFactory
import java.security.SecureRandom
import java.util.concurrent.TimeUnit
import javax.inject.Qualifier
import javax.inject.Singleton
import javax.net.ssl.SSLContext
import javax.net.ssl.TrustManagerFactory
import javax.net.ssl.X509TrustManager

@Qualifier
@Retention(AnnotationRetention.BINARY)
annotation class BootstrapRetrofit

@Qualifier
@Retention(AnnotationRetention.BINARY)
annotation class ApiRetrofit

@Module
@InstallIn(SingletonComponent::class)
object NetworkModule {

    private const val BASE_URL = "https://api.pronouns.app/"
    private const val BOOTSTRAP_URL = "https://bootstrap.pronouns.app/"
    private const val TIMEOUT_SECONDS = 30L

    @Provides
    @Singleton
    fun provideLoggingInterceptor(): HttpLoggingInterceptor {
        return HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BODY
        }
    }

    @Provides
    @Singleton
    fun provideCertificatePinner(): OkHttpCertificatePinner {
        return OkHttpCertificatePinner.Builder()
            .add("api.pronouns.app", "sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
            .add("api.pronouns.app", "sha256/BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=")
            .add("bootstrap.pronouns.app", "sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
            .add("cdn.pronouns.app", "sha256/BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=")
            .build()
    }

    @Provides
    @Singleton
    fun provideTrustManager(): X509TrustManager {
        val trustManagerFactory = TrustManagerFactory.getInstance(
            TrustManagerFactory.getDefaultAlgorithm()
        )
        trustManagerFactory.init(null as KeyStore?)
        val trustManagers = trustManagerFactory.trustManagers
        return trustManagers.first { it is X509TrustManager } as X509TrustManager
    }

    @Provides
    @Singleton
    fun provideSslContext(trustManager: X509TrustManager): SSLContext {
        val sslContext = SSLContext.getInstance("TLSv1.3")
        sslContext.init(null, arrayOf(trustManager), SecureRandom())
        return sslContext
    }

    @Provides
    @Singleton
    @BootstrapRetrofit
    fun provideBootstrapOkHttpClient(
        loggingInterceptor: HttpLoggingInterceptor,
        certificatePinner: OkHttpCertificatePinner,
        sslContext: SSLContext,
        trustManager: X509TrustManager
    ): OkHttpClient {
        return OkHttpClient.Builder()
            .connectTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .readTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .writeTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .addInterceptor(loggingInterceptor)
            .sslSocketFactory(sslContext.socketFactory, trustManager)
            .certificatePinner(certificatePinner)
            .hostnameVerifier { hostname, _ ->
                hostname.endsWith(".pronouns.app") || hostname == "bootstrap.pronouns.app"
            }
            .build()
    }

    @Provides
    @Singleton
    @ApiRetrofit
    fun provideApiOkHttpClient(
        loggingInterceptor: HttpLoggingInterceptor,
        certificatePinner: OkHttpCertificatePinner,
        authInterceptor: AuthInterceptor,
        sslContext: SSLContext,
        trustManager: X509TrustManager
    ): OkHttpClient {
        return OkHttpClient.Builder()
            .connectTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .readTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .writeTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .addInterceptor(authInterceptor)
            .addInterceptor(loggingInterceptor)
            .sslSocketFactory(sslContext.socketFactory, trustManager)
            .certificatePinner(certificatePinner)
            .hostnameVerifier { hostname, _ ->
                hostname.endsWith(".pronouns.app")
            }
            .build()
    }

    @Provides
    @Singleton
    @BootstrapRetrofit
    fun provideBootstrapRetrofit(
        @BootstrapRetrofit client: OkHttpClient
    ): Retrofit {
        return Retrofit.Builder()
            .baseUrl(BOOTSTRAP_URL)
            .client(client)
            .addConverterFactory(MoshiConverterFactory.create())
            .build()
    }

    @Provides
    @Singleton
    @ApiRetrofit
    fun provideApiRetrofit(
        @ApiRetrofit client: OkHttpClient
    ): Retrofit {
        return Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(client)
            .addConverterFactory(MoshiConverterFactory.create())
            .build()
    }

    @Provides
    @Singleton
    fun provideBootstrapApi(
        @BootstrapRetrofit retrofit: Retrofit
    ): BootstrapApi = retrofit.create(BootstrapApi::class.java)

    @Provides
    @Singleton
    fun provideDeviceApi(
        @ApiRetrofit retrofit: Retrofit
    ): DeviceApi = retrofit.create(DeviceApi::class.java)

    @Provides
    @Singleton
    fun provideUpdateApi(
        @ApiRetrofit retrofit: Retrofit
    ): UpdateApi = retrofit.create(UpdateApi::class.java)
}
