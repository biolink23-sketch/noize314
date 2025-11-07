import streamlit as st
import numpy as np
from scipy import signal
from scipy import fft
from scipy import stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
import wave
import time
import base64

# Настройка страницы
st.set_page_config(
    page_title="Генератор Розового Шума для Медитации",
    page_icon="🎵",
    layout="wide"
)

class PinkNoiseGenerator:
    """Класс для генерации розового шума с характеристикой 1/f"""
    
    def __init__(self, duration=10, sample_rate=44100):
        self.duration = duration
        self.sample_rate = sample_rate
        self.noise = None
        
    def generate_pink_noise_fft(self, alpha=1.0):
        """
        Генерация розового шума методом формирования спектра
        alpha: показатель степени (1.0 для классического розового шума 1/f)
        """
        samples = self.duration * self.sample_rate
        
        # Генерируем белый шум в частотной области
        freqs = np.fft.rfftfreq(samples, 1/self.sample_rate)
        
        # Создаем случайные фазы
        phases = np.random.uniform(0, 2*np.pi, len(freqs))
        
        # Формируем спектр с характеристикой 1/f^alpha
        # Избегаем деления на ноль для нулевой частоты
        power = np.zeros_like(freqs)
        power[0] = 1
        power[1:] = 1 / (freqs[1:] ** alpha)
        
        # Создаем комплексный спектр
        spectrum = np.sqrt(power) * np.exp(1j * phases)
        
        # Обратное преобразование Фурье
        pink_noise = np.fft.irfft(spectrum, n=samples)
        
        # Нормализация
        pink_noise = pink_noise / np.max(np.abs(pink_noise))
        
        self.noise = pink_noise
        return pink_noise
    
    def generate_pink_noise_voss(self, num_octaves=16):
        """
        Генерация розового шума алгоритмом Восса-Маккартни
        Более эффективный метод для реального времени
        """
        samples = self.duration * self.sample_rate
        pink_noise = np.zeros(samples)
        
        # Массив для хранения значений генераторов
        generators = np.zeros(num_octaves)
        
        for i in range(samples):
            # Обновляем генераторы в зависимости от позиции
            for j in range(num_octaves):
                if i % (2**j) == 0:
                    generators[j] = np.random.randn()
            
            # Суммируем все генераторы
            pink_noise[i] = np.sum(generators)
        
        # Нормализация
        pink_noise = pink_noise / np.max(np.abs(pink_noise))
        
        self.noise = pink_noise
        return pink_noise
    
    def apply_envelope(self, fade_in=2.0, fade_out=2.0):
        """Применение огибающей для плавного начала и конца"""
        if self.noise is None:
            return
        
        fade_in_samples = int(fade_in * self.sample_rate)
        fade_out_samples = int(fade_out * self.sample_rate)
        
        # Создаем огибающую
        envelope = np.ones_like(self.noise)
        
        # Плавное нарастание
        if fade_in_samples > 0:
            envelope[:fade_in_samples] = np.linspace(0, 1, fade_in_samples)
        
        # Плавное затухание
        if fade_out_samples > 0:
            envelope[-fade_out_samples:] = np.linspace(1, 0, fade_out_samples)
        
        self.noise *= envelope
    
    def get_spectrum(self):
        """Получение спектра мощности для визуализации"""
        if self.noise is None:
            return None, None
        
        # Вычисляем спектр мощности
        freqs = np.fft.rfftfreq(len(self.noise), 1/self.sample_rate)
        spectrum = np.abs(np.fft.rfft(self.noise))
        
        # Преобразуем в децибелы
        spectrum_db = 20 * np.log10(spectrum + 1e-10)
        
        return freqs[1:], spectrum_db[1:]  # Исключаем нулевую частоту
    
    def save_wav(self, volume=0.5):
        """Сохранение в WAV файл"""
        if self.noise is None:
            return None
        
        # Создаем буфер для файла
        buffer = BytesIO()
        
        # Применяем громкость
        audio_data = self.noise * volume
        
        # Конвертируем в 16-битные целые числа
        audio_data = np.int16(audio_data * 32767)
        
        # Создаем WAV файл
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Моно
            wav_file.setsampwidth(2)  # 16 бит
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        buffer.seek(0)
        return buffer
    
    def get_audio_base64(self, volume=0.5):
        """Получить base64 представление аудио для HTML5 плеера"""
        wav_buffer = self.save_wav(volume)
        if wav_buffer:
            audio_base64 = base64.b64encode(wav_buffer.read()).decode()
            wav_buffer.seek(0)
            return audio_base64
        return None

# Интерфейс Streamlit
def main():
    st.title("🎵 Генератор Розового Шума для Медитации")
    st.markdown("""
    Розовый шум (1/f шум) - это сигнал со спектральной плотностью мощности, 
    обратно пропорциональной частоте. Он широко используется для медитации, 
    улучшения сна и концентрации.
    """)
    
    # Инициализация session state
    if 'generator' not in st.session_state:
        st.session_state.generator = PinkNoiseGenerator()
    
    if 'audio_generated' not in st.session_state:
        st.session_state.audio_generated = False
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Настройки генератора")
        
        st.subheader("Параметры шума")
        
        # Метод генерации
        generation_method = st.selectbox(
            "Метод генерации",
            ["FFT метод (точный)", "Алгоритм Восса (быстрый)"],
            help="FFT метод дает более точный спектр, алгоритм Восса работает быстрее"
        )
        
        # Длительность
        duration = st.slider(
            "Длительность (секунды)",
            min_value=5,
            max_value=60,
            value=30,
            step=5,
            help="Длительность генерируемого шума (ограничено 60 сек для веб-версии)"
        )
        
        # Характеристика спектра (для FFT метода)
        if generation_method == "FFT метод (точный)":
            alpha = st.slider(
                "Показатель спектра (α)",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="1.0 - классический розовый шум (1/f), 0 - белый шум, 2 - броуновский шум"
            )
        else:
            num_octaves = st.slider(
                "Количество октав",
                min_value=8,
                max_value=24,
                value=16,
                step=1,
                help="Больше октав - более точный розовый шум"
            )
        
        st.subheader("Огибающая")
        
        # Параметры огибающей
        use_envelope = st.checkbox("Использовать огибающую", value=True)
        
        if use_envelope:
            fade_in = st.slider(
                "Плавное нарастание (сек)",
                min_value=0.0,
                max_value=5.0,
                value=2.0,
                step=0.5
            )
            
            fade_out = st.slider(
                "Плавное затухание (сек)",
                min_value=0.0,
                max_value=5.0,
                value=2.0,
                step=0.5
            )
        
        st.subheader("Громкость")
        volume = st.slider(
            "Уровень громкости",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        # Частота дискретизации
        sample_rate = st.selectbox(
            "Частота дискретизации (Гц)",
            [22050, 44100, 48000],
            index=1,
            help="Качество звука (44100 Гц - CD качество)"
        )
    
    # Основной контент
    if st.button("🎲 Генерировать розовый шум", type="primary", use_container_width=True):
        with st.spinner("Генерация розового шума..."):
            # Обновляем параметры генератора
            st.session_state.generator = PinkNoiseGenerator(duration, sample_rate)
            
            # Генерируем шум выбранным методом
            if generation_method == "FFT метод (точный)":
                st.session_state.generator.generate_pink_noise_fft(alpha)
            else:
                st.session_state.generator.generate_pink_noise_voss(num_octaves)
            
            # Применяем огибающую если нужно
            if use_envelope:
                st.session_state.generator.apply_envelope(fade_in, fade_out)
            
            st.session_state.audio_generated = True
            st.success("✅ Розовый шум успешно сгенерирован!")
    
    # Воспроизведение и скачивание
    if st.session_state.audio_generated and st.session_state.generator.noise is not None:
        
        st.header("🎧 Воспроизведение")
        
        # Получаем audio в base64 для HTML5 плеера
        audio_base64 = st.session_state.generator.get_audio_base64(volume)
        
        if audio_base64:
            # HTML5 аудио плеер
            audio_html = f"""
            <audio controls autoplay>
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                Ваш браузер не поддерживает элемент audio.
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
            
            st.info(f"📢 Длительность: {duration} секунд | Громкость: {int(volume*100)}%")
        
        # Визуализация
        st.header("📊 Визуализация")
        
        tab1, tab2, tab3 = st.tabs(["Форма волны", "Спектр мощности", "Спектрограмма"])
        
        with tab1:
            # Показываем первые несколько секунд волны
            samples_to_show = min(st.session_state.generator.sample_rate * 2, 
                                 len(st.session_state.generator.noise))
            time_axis = np.linspace(0, samples_to_show/st.session_state.generator.sample_rate, 
                                  samples_to_show)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=st.session_state.generator.noise[:samples_to_show],
                mode='lines',
                name='Розовый шум',
                line=dict(color='pink', width=0.5)
            ))
            fig.update_layout(
                title="Форма волны (первые 2 секунды)",
                xaxis_title="Время (сек)",
                yaxis_title="Амплитуда",
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            freqs, spectrum_db = st.session_state.generator.get_spectrum()
            if freqs is not None:
                # Ограничиваем частоты для лучшей визуализации
                mask = (freqs > 10) & (freqs < 5000)
                
                fig = go.Figure()
                
                # Спектр розового шума
                fig.add_trace(go.Scatter(
                    x=freqs[mask],
                    y=spectrum_db[mask],
                    mode='lines',
                    name='Спектр мощности',
                    line=dict(color='blue', width=1)
                ))
                
                # Теоретическая линия 1/f
                if generation_method == "FFT метод (точный)":
                    theoretical = -10 * alpha * np.log10(freqs[mask]/freqs[mask][0])
                    theoretical = theoretical - np.mean(theoretical) + np.mean(spectrum_db[mask])
                    
                    fig.add_trace(go.Scatter(
                        x=freqs[mask],
                        y=theoretical,
                        mode='lines',
                        name=f'Теоретический 1/f^{alpha:.1f}',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    title="Спектр мощности (логарифмические координаты)",
                    xaxis_title="Частота (Гц)",
                    yaxis_title="Мощность (дБ)",
                    xaxis_type="log",
                    height=400,
                    showlegend=True,
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Показываем наклон спектра
                log_freqs = np.log10(freqs[mask])
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    log_freqs, spectrum_db[mask]
                )
                
                expected_slope = -10 * (alpha if generation_method == "FFT метод (точный)" else 1.0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Измеренный наклон", f"{slope:.2f} дБ/декада")
                with col2:
                    st.metric("Теоретический наклон", f"{expected_slope:.1f} дБ/декада")
                with col3:
                    st.metric("Коэффициент корреляции", f"{r_value**2:.3f}")
        
        with tab3:
            # Спектрограмма
            f, t, Sxx = signal.spectrogram(
                st.session_state.generator.noise, 
                st.session_state.generator.sample_rate,
                nperseg=1024,
                noverlap=512
            )
            
            # Ограничиваем частоты
            freq_mask = f < 5000
            f_limited = f[freq_mask]
            Sxx_limited = Sxx[freq_mask, :]
            
            fig = go.Figure(data=go.Heatmap(
                x=t,
                y=f_limited,
                z=10 * np.log10(Sxx_limited + 1e-10),
                colorscale='Viridis',
                colorbar=dict(title="Мощность (дБ)"),
                zmin=-60,
                zmax=0
            ))
            
            fig.update_layout(
                title="Спектрограмма",
                xaxis_title="Время (сек)",
                yaxis_title="Частота (Гц)",
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Скачивание
        st.header("💾 Сохранение")
        
        col1, col2 = st.columns(2)
        
        with col1:
            wav_buffer = st.session_state.generator.save_wav(volume)
            if wav_buffer:
                st.download_button(
                    label="📥 Скачать WAV файл",
                    data=wav_buffer,
                    file_name=f"pink_noise_{int(time.time())}.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
        
        with col2:
            # Информация о файле
            st.info(f"""
            **Информация о файле:**
            - Формат: WAV (несжатый)
            - Длительность: {duration} сек
            - Частота: {sample_rate} Гц
            - Разрядность: 16 бит
            - Каналы: Моно
            - Размер: ~{duration * sample_rate * 2 / 1024 / 1024:.1f} МБ
            """)
    
    # Информационная секция
    with st.expander("ℹ️ О розовом шуме и его применении"):
        st.markdown("""
        ### Что такое розовый шум?
        
        Розовый шум (также известный как 1/f шум или фликкер-шум) - это сигнал, 
        спектральная плотность мощности которого обратно пропорциональна частоте:
        
        **W(f) = W(1) / f^α**
        
        где α ≈ 1 для классического розового шума.
        
        ### Почему розовый шум эффективен для медитации?
        
        1. **Естественность**: Розовый шум встречается во многих природных процессах:
           - Шум водопада
           - Морской прибой  
           - Шелест листьев
           - Сердечный ритм
        
        2. **Баланс частот**: В отличие от белого шума, розовый шум имеет больше энергии 
           на низких частотах, что делает его более приятным для восприятия
        
        3. **Научные исследования**: Показано, что розовый шум:
           - Улучшает глубокий сон
           - Способствует консолидации памяти
           - Снижает время засыпания
           - Маскирует отвлекающие звуки
        
        ### Рекомендации по использованию:
        
        - **Для медитации**: 10-20 минут, громкость 30-40%
        - **Для сна**: 30+ минут с таймером выключения, громкость 20-30%
        - **Для концентрации**: Непрерывно во время работы, громкость 25-35%
        - **Начинайте с низкой громкости** и постепенно увеличивайте до комфортного уровня
        - **Используйте качественные наушники или колонки** для лучшего эффекта
        
        ### Типы шума по спектру:
        
        - **Белый шум** (f^0): Равная мощность на всех частотах
        - **Розовый шум** (1/f): Мощность падает на 3 дБ на октаву
        - **Коричневый шум** (1/f²): Мощность падает на 6 дБ на октаву
        
        Розовый шум находится посередине и часто воспринимается как наиболее 
        естественный и комфортный для длительного прослушивания.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*🧘 Создано для практики медитации и улучшения концентрации*")
    st.markdown("*Используйте розовый шум ответственно и не превышайте безопасный уровень громкости*")

if __name__ == "__main__":
    main()
