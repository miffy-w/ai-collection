/**
 * 使用 DeepSeek API 定时更新 README 文档
 */
const axios = require('axios');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

// DeepSeek API 配置
const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY || 'your_api_key_here';
const DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions';

// 项目路径
const PROJECT_DIR = path.resolve(__dirname);
const README_CN = path.join(PROJECT_DIR, 'README.md');
const README_EN = path.join(PROJECT_DIR, 'README_EN.md');

/**
 * 生成更新提示词
 */
function generateUpdatePrompt(content, isEnglish = false) {
    const currentDate = new Date().toLocaleDateString('zh-CN', { year: 'numeric', month: 'long' });
    const currentDateEn = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long' });

    if (isEnglish) {
        return `As an AI resource collection expert, update the following AI tools and platforms list:

1. Check if tools in the list are still available/active
2. Add new popular AI tools (2025-2026), If there is one...
3. Remove tools that are closed or no longer maintained
4. Keep the existing format and structure
5. Update the last updated date to: ${currentDateEn}

Current content:
${content}

Please return the complete updated content with the same Markdown format.`;
    } else {
        return `请作为 AI 资源收集专家，更新以下 AI 工具和平台列表：

1. 检查列表中的工具是否仍然可用/活跃
2. 添加最近流行的 AI 工具（2025-2026年），如果有的话
3. 移除已关闭或不再维护的工具
4. 保持现有的格式和结构
5. 更新最后更新日期为：${currentDate}

现有内容：
${content}

请返回完整的更新后的内容，保持相同的 Markdown 格式。`;
    }
}

/**
 * 调用 DeepSeek API
 */
async function callDeepSeekAPI(prompt) {
    const headers = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${DEEPSEEK_API_KEY}`
    };

    const data = {
        model: 'deepseek-chat',
        messages: [
            {
                role: 'system',
                content: isEnglish => isEnglish 
                    ? 'You are a professional AI resource collection and organization expert, skilled at maintaining lists of AI tools and platforms.'
                    : '你是一个专业的 AI 资源收集和整理专家，擅长维护 AI 工具和平台的列表。'
            },
            {
                role: 'user',
                content: prompt
            }
        ],
        temperature: 0.7,
        max_tokens: 16000
    };

    try {
        const response = await axios.post(DEEPSEEK_API_URL, data, {
            headers,
            timeout: 60000
        });
        return response.data.choices[0].message.content;
    } catch (error) {
        console.error('API 调用失败:', error.response?.data || error.message);
        return null;
    }
}

/**
 * 更新 README 文件
 */
async function updateReadmeFile(filePath, isEnglish = false) {
    console.log(`正在更新: ${filePath}`);

    try {
        // 读取现有内容
        const currentContent = fs.readFileSync(filePath, 'utf-8');

        // 生成提示词
        const prompt = generateUpdatePrompt(currentContent, isEnglish);

        // 调用 DeepSeek API
        console.log('正在调用 DeepSeek API...');
        const updatedContent = await callDeepSeekAPI(prompt);

        if (updatedContent) {
            // 写入更新后的内容
            fs.writeFileSync(filePath, updatedContent, 'utf-8');
            console.log(`✓ ${path.basename(filePath)} 更新成功`);
            return true;
        } else {
            console.log(`✗ ${path.basename(filePath)} 更新失败：API 无响应`);
            return false;
        }
    } catch (error) {
        console.log(`✗ ${path.basename(filePath)} 更新失败:`, error.message);
        return false;
    }
}

/**
 * 主函数
 */
async function main() {
    console.log('='.repeat(50));
    console.log('开始更新 README 文档');
    console.log(`时间: ${new Date().toLocaleString('zh-CN')}`);
    console.log('='.repeat(50));

    // 更新中文 README
    const successCN = await updateReadmeFile(README_CN, false);

    // 更新英文 README
    const successEN = await updateReadmeFile(README_EN, true);

    console.log('='.repeat(50));
    if (successCN && successEN) {
        console.log('✓ 所有文档更新完成');
    } else if (successCN || successEN) {
        console.log('⚠ 部分文档更新完成');
    } else {
        console.log('✗ 所有文档更新失败');
    }
    console.log('='.repeat(50));

    // 返回退出码
    process.exit(successCN || successEN ? 0 : 1);
}

// 如果直接运行此文件，执行主函数
if (require.main === module) {
    main().catch(error => {
        console.error('发生错误:', error);
        process.exit(1);
    });
}

module.exports = { updateReadmeFile };