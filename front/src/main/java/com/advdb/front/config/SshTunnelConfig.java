package com.advdb.front.config;

import com.jcraft.jsch.JSch;
import com.jcraft.jsch.JSchException;
import com.jcraft.jsch.Session;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Configuration;
import java.io.Console;
import java.util.Properties;
import java.util.Scanner;

@Configuration
@ConditionalOnProperty(name = "ssh.tunnel.enabled", havingValue = "true")
public class SshTunnelConfig {

    private static final Logger logger = LoggerFactory.getLogger(SshTunnelConfig.class);

    @Value("${ssh.tunnel.host}")
    private String sshHost;

    @Value("${ssh.tunnel.port}")
    private int sshPort;

    @Value("${ssh.tunnel.username}")
    private String sshUsername;

    @Value("${ssh.tunnel.password:}")
    private String sshPassword;

    @Value("${ssh.tunnel.local-port}")
    private int localPort;

    @Value("${ssh.tunnel.remote-host}")
    private String remoteHost;

    @Value("${ssh.tunnel.remote-port}")
    private int remotePort;

    private Session session;

    @PostConstruct
    public void startTunnel() {
        try {
            logger.info("tunnel SSH vers {}", sshPort);
            
            JSch jsch = new JSch();
            session = jsch.getSession(sshUsername, sshHost, sshPort);
            
            session.setPassword(sshPassword);
            
            Properties config = new Properties();
            config.put("StrictHostKeyChecking", "no");
            session.setConfig(config);
            
            session.connect();
            logger.info("connexion ssh établie");
            
            session.setPortForwardingL(localPort, remoteHost, remotePort);
            logger.info("localhost:{} -> {}:{} via {}", localPort, remoteHost, remotePort, sshPort);
            
        } catch (JSchException e) {
            logger.error("erreur", e);
            throw new RuntimeException("state:", e);
        }
    }

    @PreDestroy
    public void stopTunnel() {
        if (session != null && session.isConnected()) {
            logger.info("fermeture du tunnel");
            session.disconnect();
        }
    }
}
