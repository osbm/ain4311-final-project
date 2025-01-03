#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "esp_camera.h"
#include <UniversalTelegramBot.h>
#include <ArduinoJson.h>

#include "model.h"  // Include your TFLite model as a C array
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


const char* ssid_osbm = "AAAAAAAA";
const char* password_osbm = "AAAAAAA";

const char* ssid_aleyna = "AAAAAA";
const char* password_aleyna= "AAAAAAAAA";

w
// Initialize Telegram BOT
#define BOTtoken "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"  // your Bot Token (Get from Botfather)

// Use @myidbot to find out the chat ID of an individual or a group
// Also note that you need to click "start" on a bot before it can
// message you
#define CHAT_ID "AAAAAAAAAA"

bool sendPhoto = false;

WiFiClientSecure clientTCP;
UniversalTelegramBot bot(BOTtoken, clientTCP);

#define FLASH_LED_PIN 4
bool flashState = LOW;

//Checks for new messages every 1 second.
int botRequestDelay = 1000;
unsigned long lastTimeBotRan;




// Constants
const int tensor_arena_size = 10 * 1024;  // Adjust based on model size
uint8_t tensor_arena[tensor_arena_size];

// TensorFlow Lite variables
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input_tensor;
TfLiteTensor* output_tensor;

//CAMERA_MODEL_AI_THINKER
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22


void configInitCamera(){
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;

  //init with high specs to pre-allocate larger buffers
  if(psramFound()){
    config.frame_size = FRAMESIZE_UXGA;
    config.jpeg_quality = 10;  //0-63 lower number means higher quality
    config.fb_count = 1;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;  //0-63 lower number means higher quality
    config.fb_count = 1;
  }

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    delay(1000);
    ESP.restart();
  }
}

void handleNewMessages(int numNewMessages) {
  Serial.print("Handle New Messages: ");
  Serial.println(numNewMessages);

  for (int i = 0; i < numNewMessages; i++) {
    String chat_id = String(bot.messages[i].chat_id);
    if (chat_id != CHAT_ID){
      bot.sendMessage(chat_id, "Unauthorized user", "");
      continue;
    }

    // Print the received message
    String text = bot.messages[i].text;
    Serial.println(text);

    String from_name = bot.messages[i].from_name;
    if (text == "/start") {
      String welcome = "Welcome , " + from_name + "\n";
      welcome += "Use the following commands to interact with the ESP32-CAM \n";
      welcome += "/photo : takes a new photo\n";
      welcome += "/flash : toggles flash LED \n";
      bot.sendMessage(CHAT_ID, welcome, "");
    }
    if (text == "/flash") {
      flashState = !flashState;
      digitalWrite(FLASH_LED_PIN, flashState);
      Serial.println("Change flash LED state");
    }
    if (text == "/photo") {
      sendPhoto = true;

      Serial.println("New photo request");
    }
  }
}

String sendPhotoTelegram() {
  const char* myDomain = "api.telegram.org";
  String getAll = "";
  String getBody = "";

  //Dispose first picture because of bad quality
  camera_fb_t * fb = NULL;
  fb = esp_camera_fb_get();
  esp_camera_fb_return(fb); // dispose the buffered image

  // Take a new photo
  fb = NULL;
  fb = esp_camera_fb_get();
  if(!fb) {
    Serial.println("Camera capture failed");
    delay(1000);
    ESP.restart();
    return "Camera capture failed";
  }

  Serial.println("Connect to " + String(myDomain));


  if (clientTCP.connect(myDomain, 443)) {
    Serial.println("Connection successful");

    String head = "--RandomNerdTutorials\r\nContent-Disposition: form-data; name=\"chat_id\"; \r\n\r\n"  CHAT_ID  "\r\n--RandomNerdTutorials\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"esp32-cam.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n";
    String tail = "\r\n--RandomNerdTutorials--\r\n";

    size_t imageLen = fb->len;
    size_t extraLen = head.length() + tail.length();
    size_t totalLen = imageLen + extraLen;

    clientTCP.println("POST /bot" BOTtoken "/sendPhoto HTTP/1.1");
    clientTCP.println("Host: " + String(myDomain));
    clientTCP.println("Content-Length: " + String(totalLen));
    clientTCP.println("Content-Type: multipart/form-data; boundary=RandomNerdTutorials");
    clientTCP.println();
    clientTCP.print(head);

    uint8_t *fbBuf = fb->buf;
    size_t fbLen = fb->len;
    for (size_t n=0;n<fbLen;n=n+1024) {
      if (n+1024<fbLen) {
        clientTCP.write(fbBuf, 1024);
        fbBuf += 1024;
      }
      else if (fbLen%1024>0) {
        size_t remainder = fbLen%1024;
        clientTCP.write(fbBuf, remainder);
      }
    }

    clientTCP.print(tail);

    esp_camera_fb_return(fb);

    int waitTime = 10000;   // timeout 10 seconds
    long startTimer = millis();
    boolean state = false;

    while ((startTimer + waitTime) > millis()){
      Serial.print(".");
      delay(100);
      while (clientTCP.available()) {
        char c = clientTCP.read();
        if (state==true) getBody += String(c);
        if (c == '\n') {
          if (getAll.length()==0) state=true;
          getAll = "";
        }
        else if (c != '\r')
          getAll += String(c);
        startTimer = millis();
      }
      if (getBody.length()>0) break;
    }
    clientTCP.stop();
    Serial.println(getBody);
  }
  else {
    getBody="Connected to api.telegram.org failed.";
    Serial.println("Connected to api.telegram.org failed.");
  }
  return getBody;
}

void setup(){
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  // Init Serial Monitor
  Serial.begin(115200);

  // Set LED Flash as output
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, flashState);

  // Config and init the camera
  configInitCamera();
  int patience;
  // Connect to Wi-Fi
  WiFi.mode(WIFI_STA);
  Serial.println();
  while (true) {
    Serial.print("Connecting to osman wifi ");
    Serial.println(ssid_osbm);
    WiFi.begin(ssid_osbm, password_osbm);
    clientTCP.setCACert(TELEGRAM_CERTIFICATE_ROOT); // Add root certificate for api.telegram.org
    patience = 10;
    while (WiFi.status() != WL_CONNECTED) {
      Serial.print(".");
      delay(500);
      patience -= 1;
      if (patience == 0) {
        break;
      }
    }
    if (patience > 0 ) {
      Serial.println();
      Serial.print("ESP32-CAM IP Address: ");
      Serial.println(WiFi.localIP());
      break;
    }

    Serial.print("Connecting to aleyna wifi ");
    Serial.println(ssid_aleyna);
    WiFi.begin(ssid_aleyna, password_aleyna);
    clientTCP.setCACert(TELEGRAM_CERTIFICATE_ROOT); // Add root certificate for api.telegram.org
    patience = 10;
    while (WiFi.status() != WL_CONNECTED) {
      Serial.print(".");
      delay(500);
      patience -= 1;
      if (patience == 0) {
        break;
      }
    }
    if (patience > 0 ) {
      Serial.println();
      Serial.print("ESP32-CAM IP Address: ");
      Serial.println(WiFi.localIP());
      break;
    }
  }


  // Load the TFLite model
  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1);
  }

  // AllOpsResolver includes all TFLite operators (can optimize for size)
  static tflite::AllOpsResolver resolver;

  // Create an interpreter
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, tensor_arena_size, nullptr);
  interpreter = &static_interpreter;

  // Allocate memory for the model
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed!");
    while (1);
  }

  // Get input and output tensors
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  // Print input and output tensor details
  Serial.println("Model loaded successfully!");
  Serial.printf("Input tensor size: %d\n", input_tensor->bytes);
  Serial.printf("Output tensor size: %d\n", output_tensor->bytes);



}
void resizeImage(camera_fb_t *fb, uint16_t new_width, uint16_t new_height, uint8_t *resized_buffer) {
  // Resize the image
  uint16_t old_width = fb->width;
  uint16_t old_height = fb->height;
  uint8_t *old_buffer = fb->buf;

  // Calculate the scaling factor
  float x_ratio = (float)old_width / new_width;
  float y_ratio = (float)old_height / new_height;

  // Resize the image
  for (uint16_t y = 0; y < new_height; y++) {
    for (uint16_t x = 0; x < new_width; x++) {
      uint16_t px = x * x_ratio;
      uint16_t py = y * y_ratio;
      uint16_t old_offset = 2 * (px + py * old_width);
      uint16_t new_offset = 2 * (x + y * new_width);
      resized_buffer[new_offset] = old_buffer[old_offset];
      resized_buffer[new_offset + 1] = old_buffer[old_offset + 1];
    }
  }
}

void loop() {
  if (sendPhoto) {
    Serial.println("Preparing photo");
    sendPhotoTelegram();

    bot.sendMessage(CHAT_ID, "processing the image", "");

    camera_fb_t * fb = NULL;

  //Dispose first picture because of bad quality
  camera_fb_t * fb = NULL;
  fb = esp_camera_fb_get();
  esp_camera_fb_return(fb); // dispose the buffered image

  // Take a new photo
  fb = NULL;
  fb = esp_camera_fb_get();
  if(!fb) {
    Serial.println("Camera capture failed");
    delay(1000);
    ESP.restart();
    return "Camera capture failed";
  }

  // Resize the image
  uint16_t new_width = 120; // Example resized width
  uint16_t new_height = 120; // Example resized height
  uint8_t resized_buffer[new_width * new_height * 3]; // Assuming RGB format
  resizeImage(fb, new_width, new_height, resized_buffer);


  // Preprocess the image
  preprocessImage(resized_buffer, new_width, new_height);

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Model invocation failed!");
    return;
  }

  // Process the output
  Serial.println("Inference output:");
  for (int i = 0; i < output_tensor->bytes; i++) {
    // 0 for dog
    // 1 for cat
    // 2 for human male
    // 3 for human female
    if (output_tensor->data.uint8[i] == 0) {
      bot.sendMessage(CHAT_ID, "class: dog", "");
    } else if (output_tensor->data.uint8[i] == 1) {
      bot.sendMessage(CHAT_ID, "class: cat", "");
    } else if (output_tensor->data.uint8[i] == 2) {
      bot.sendMessage(CHAT_ID, "class: man", "");
    } else if (output_tensor->data.uint8[i] == 3) {
      bot.sendMessage(CHAT_ID, "class: woman", "");
    };
  };

  if (millis() > lastTimeBotRan + botRequestDelay)  {
    int numNewMessages = bot.getUpdates(bot.last_message_received + 1);
    while (numNewMessages) {
      Serial.println("got response");
      handleNewMessages(numNewMessages);
      numNewMessages = bot.getUpdates(bot.last_message_received + 1);
    }
    lastTimeBotRan = millis();
  }
}