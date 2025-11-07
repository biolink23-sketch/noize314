import streamlit as st
import numpy as np
import scipy.signal as signal
from scipy import fft
import sounddevice as sd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
import wave
import struct
import time

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
        self.is_playing = False
        
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
        
    def play_noise(self, volume=0.5):
        """Воспроизведение сгенерированного шума"""
        if self.noise is not None:
            sd.play(self.noise * volume, self.sample_rate)
            self.is_playing = True
    
    def stop_noise(self):
        """Остановка воспроизведения"""
        sd.stop()
        self.is_playing = False
    
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
    
    def save_wav(self):
        """Сохранение в WAV файл"""
        if self.noise is None:
            return None
        
        # Создаем буфер для файла
        buffer = BytesIO()
        
        # Конвертируем в 16-битные целые числа
        audio_data = np.int16(self.noise * 32767)
        
        # Создаем WAV файл
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Моно
            wav_file.setsampwidth(2)  # 16 бит
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        buffer.seek(0)
        return buffer

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
    
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False
    
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
            max_value=300,
            value=30,
            step=5,
            help="Длительность генерируемого шума"
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
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🎲 Генерировать шум", type="primary", use_container_width=True):
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
                
                st.success("Розовый шум успешно сгенерирован!")
    
    with col2:
        if st.button("▶️ Воспроизвести", use_container_width=True, 
                     disabled=st.session_state.generator.noise is None):
            if not st.session_state.is_playing:
                st.session_state.generator.play_noise(volume)
                st.session_state.is_playing = True
                st.info(f"Воспроизведение {duration} секунд...")
            else:
                st.warning("Уже воспроизводится!")
    
    with col3:
        if st.button("⏹️ Остановить", use_container_width=True):
            st.session_state.generator.stop_noise()
            st.session_state.is_playing = False
            st.info("Воспроизведение остановлено")
    
    # Визуализация
    if st.session_state.generator.noise is not None:
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
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            freqs, spectrum_db = st.session_state.generator.get_spectrum()
            if freqs is not None:
                # Ограничиваем частоты для лучшей визуализации
                mask = freqs < 5000
                
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
                theoretical = -10 * np.log10(freqs[mask])
                theoretical = theoretical - np.mean(theoretical) + np.mean(spectrum_db[mask])
                
                fig.add_trace(go.Scatter(
                    x=freqs[mask],
                    y=theoretical,
                    mode='lines',
                    name='Теоретический 1/f',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title="Спектр мощности (логарифмические координаты)",
                    xaxis_title="Частота (Гц)",
                    yaxis_title="Мощность (дБ)",
                    xaxis_type="log",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Показываем наклон спектра
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    np.log10(freqs[mask]), spectrum_db[mask]
                )
                st.info(f"Наклон спектра: {slope:.2f} дБ/декада (теоретический для розового шума: -10 дБ/декада)")
        
        with tab3:
            # Спектрограмма
            f, t, Sxx = signal.spectrogram(
                st.session_state.generator.noise, 
                st.session_state.generator.sample_rate,
                nperseg=1024
            )
            
            fig = go.Figure(data=go.Heatmap(
                x=t,
                y=f,
                z=10 * np.log10(Sxx + 1e-10),
                colorscale='Viridis',
                colorbar=dict(title="Мощность (дБ)")
            ))
            
            fig.update_layout(
                title="Спектрограмма",
                xaxis_title="Время (сек)",
                yaxis_title="Частота (Гц)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Скачивание
        st.header("💾 Сохранение")
        
        col1, col2 = st.columns(2)
        
        with col1:
            wav_buffer = st.session_state.generator.save_wav()
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
            """)
    
    # Информационная секция
    with st.expander("ℹ️ О розовом шуме и его применении"):
        st.markdown("""
        ### Что такое розовый шум?
        
        Розовый шум (также известный как 1/f шум или фликкер-шум) - это сигнал, 
        спектральная плотность мощности которого обратно пропорциональна частоте:
        
        **W(f) = W(1) / f^α**
        
        где α ≈ 1 для классического розового шума.
        
        ### Преимущества для медитации:
        
        1. **Естественность**: Розовый шум встречается во многих природных процессах
        2. **Маскировка**: Эффективно маскирует отвлекающие звуки
        3. **Расслабление**: Способствует глубокому расслаблению и медитации
        4. **Улучшение сна**: Помогает быстрее заснуть и улучшает качество сна
        5. **Концентрация**: Улучшает фокусировку внимания
        
        ### Рекомендации по использованию:
        
        - Начните с низкой громкости и постепенно увеличивайте до комфортного уровня
        - Используйте качественные наушники или колонки
        - Экспериментируйте с разными длительностями для разных целей
        - Для сна рекомендуется более длительное воспроизведение (30+ минут)
        - Для коротких медитаций достаточно 5-10 минут
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*Создано для практики медитации и улучшения концентрации* 🧘")

if __name__ == "__main__":
    main()
