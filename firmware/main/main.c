// ESP32-C3 CSI sensor
//
// 1. Connect to the phone hotspot.
// 2. Ping the hotspot 100 times per second, so it keeps sending us packets.
// 3. For every packet from the hotspot, print its CSI as one line over USB:
//      CSI_DATA,<rssi>,<len>,[v0,v1,v2,...]
//    The numbers come in pairs (imaginary, real), one pair per subcarrier.

#include <stdio.h>
#include <string.h>
#include "nvs_flash.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "ping/ping_sock.h"

#define WIFI_SSID "aaa"
#define WIFI_PASSWORD "aa"
#define PINGS_PER_SECOND 100

static const char *TAG = "csi";
static uint8_t hotspot_mac[6]; // we only keep packets sent by this address

// Called by the Wi-Fi driver for every packet it receives.
static void csi_callback(void *ctx, wifi_csi_info_t *info)
{
    if (memcmp(info->mac, hotspot_mac, 6) != 0)
    {
        return; // packet from some other device, ignore
    }

    // Build the whole line first, then print it in one go.
    static char line[1024];
    int n = sprintf(line, "CSI_DATA,%d,%d,[", info->rx_ctrl.rssi, info->len);
    for (int i = 0; i < info->len; i++)
    {
        n += sprintf(line + n, i == 0 ? "%d" : ",%d", info->buf[i]);
    }
    sprintf(line + n, "]\n");
    fputs(line, stdout);
}

// Turn on CSI for packets coming from the hotspot.
static void start_csi(void)
{
    wifi_ap_record_t ap;
    esp_wifi_sta_get_ap_info(&ap);
    memcpy(hotspot_mac, ap.bssid, 6);

    wifi_csi_config_t csi_config = {
        .lltf_en = true, // use the LLTF part of the packet (works with any access point)
        .htltf_en = false,
        .stbc_htltf2_en = false,
        .ltf_merge_en = true,
        .channel_filter_en = true,
        .manu_scale = false,
        .shift = 0,
    };
    esp_wifi_set_csi_config(&csi_config);
    esp_wifi_set_csi_rx_cb(csi_callback, NULL);
    esp_wifi_set_csi(true);
}

// Ping the hotspot forever. Every reply it sends back gives us one CSI line.
static void start_ping(esp_ip4_addr_t hotspot_ip)
{
    esp_ping_config_t config = ESP_PING_DEFAULT_CONFIG();
    config.target_addr.type = ESP_IPADDR_TYPE_V4;
    config.target_addr.u_addr.ip4.addr = hotspot_ip.addr;
    config.count = 0; // 0 = never stop
    config.interval_ms = 1000 / PINGS_PER_SECOND;
    config.data_size = 1; // tiny packets

    esp_ping_callbacks_t callbacks = {0}; // we don't need ping results
    esp_ping_handle_t ping;
    esp_ping_new_session(&config, &callbacks, &ping);
    esp_ping_start(ping);
}

// Reacts to Wi-Fi events: started, disconnected, got an IP address.
static void on_wifi_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    static bool started = false;

    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START)
    {
        esp_wifi_connect();
    }
    else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED)
    {
        wifi_event_sta_disconnected_t *event = data;
        ESP_LOGW(TAG, "Not connected (reason %d), trying again...", event->reason);
        esp_wifi_connect(); // keep trying forever
    }
    else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP)
    {
        ip_event_got_ip_t *event = data;
        ESP_LOGI(TAG, "Connected! My IP " IPSTR ", hotspot " IPSTR,
                 IP2STR(&event->ip_info.ip), IP2STR(&event->ip_info.gw));
        if (!started)
        { // after a reconnect, CSI and ping are already running
            start_csi();
            start_ping(event->ip_info.gw); // the hotspot is our gateway
            started = true;
        }
    }
}

void app_main(void)
{
    // Standard Wi-Fi setup
    nvs_flash_init();
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();
    wifi_init_config_t init_config = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&init_config);

    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, on_wifi_event, NULL);
    esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, on_wifi_event, NULL);

    // Which network to join
    wifi_config_t wifi_config = {0};
    strcpy((char *)wifi_config.sta.ssid, WIFI_SSID);
    strcpy((char *)wifi_config.sta.password, WIFI_PASSWORD);
    wifi_config.sta.sae_pwe_h2e = WPA3_SAE_PWE_BOTH; // also works if the hotspot uses WPA3

    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    esp_wifi_start();
    esp_wifi_set_ps(WIFI_PS_NONE); // keep the radio always on, so we don't miss packets

    ESP_LOGI(TAG, "Connecting to %s...", WIFI_SSID);
}
